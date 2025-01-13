"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from pytorch_forecasting.models import BaseModel
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P - self.S),
            mode='replicate'
        )  # [B*M, 1, L] -> [B*M, 1, L+P-S]

        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]

        return x_emb  # x_emb: [B, M, D, N]


class MultiDWConv(nn.Module):
    def __init__(self, M, D, kernel_sizes=[5, 51]):
        super(MultiDWConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M * D,
                out_channels=M * D,
                kernel_size=kernel_size,
                groups=M * D,
                padding='same'
            ) for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [conv(x) for conv in self.convs]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        groups_num = M if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=M * D,
            out_channels=r * M * D,
            kernel_size=1,
            groups=groups_num
        )
        self.pw_con2 = nn.Conv1d(
            in_channels=r * M * D,
            out_channels=M * D,
            kernel_size=1,
            groups=groups_num
        )

    def forward(self, x):
        # x: [B, M*D, N]
        # 先对M维度做逐点卷积,再对D维度做逐点卷积
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
        return x  # x: [B, M*D, N]


class ModernTCNBlock(nn.Module):
    def __init__(self, M, D, kernel_sizes, r):
        super(ModernTCNBlock, self).__init__()
        # 大核逐深度可分离卷积负责捕获时域关系,卷积核大小为51,5
        self.dw_conv = MultiDWConv(M, D, kernel_sizes)
        self.bn = nn.BatchNorm1d(M * D)
        self.conv_ffn1 = ConvFFN(M, D, r, one=True)
        self.conv_ffn2 = ConvFFN(M, D, r, one=False)
        self.M=M
        self.D=D
    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]             # torch.Size([100, 11, 160, 4])
        x = rearrange(x_emb, 'b m d n -> b (m d) n')  # [B, M, D, N] -> [B, M*D, N]
        x = self.dw_conv(x)  # [B, M*D, N] -> [B, M*D, N]       # DWconv用来建模时间关系
        x = self.bn(x)  # [B, M*D, N] -> [B, M*D, N]
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_ffn1(x)  # [B, M*D, N] -> [B, M*D, N]         # ConvFFN用来建模通道关系

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)  # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0, 2, 1, 3)  # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')  # [B, D, M, N] -> [B, D*M, N]

        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_ffn2(x)  # [B, D*M, N] -> [B, D*M, N]         # ConvFFN用来建模变量关系

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)  # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0, 2, 1, 3)  # [B, D, M, N] -> [B, M, D, N]

        out = x + x_emb

        return out  # out: [B, M, D, N]


class ModernTCN(nn.Module):
    # 在顶会论文中, kernel_size=51, 5, 原来r=8,num_layers=3
    # 对于大数据集,r=8;对于ETTh1,ETTh2,Exchange,ILI等小数据集,r=1;r在1,2,4,8中选取
    # 对于大数据集,num_layers=3;对于小数据集,num_layers=1,num_layers在1,2,3中选取
    # D默认等于64,D在32,64,128中选取
    def __init__(self, M=11, L=8, T=1, D=64, P=4, S=2, kernel_sizes=[5, 51], r=1, num_layers=1):
        super(ModernTCN, self).__init__()

        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D)
        self.backbone = nn.ModuleList([ModernTCNBlock(M, D, kernel_sizes, r) for _ in range(num_layers)])
        self.head = nn.Linear(M * D * N , T)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, M]
        x = x.permute(0, 2, 1)  # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]

        # Flatten
        z = rearrange(x_emb, 'b m d n -> b (m d n)')  # [B, M, D, N] -> [B, M*D*N]
        pred = self.head(z)  # [B, M*D*N] -> [B, T]

        return pred  # out: [B, T]


class ModernTCNModel(BaseModel):
    def __init__(self,
                 enc_in:int,
                 input_size:int,
                 output_size:int,
                 hidden_size:int,
                 P=4,
                 S=2,
                 kernel_sizes=[5,51], 
                 r=2, 
                 num_layers=2,
                #  P=4,
                #  S=2,
                #  kernel_sizes=[5,51], 
                #  r=1, 
                #  num_layers=1,
                 **kwargs):

        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        # super().__init__()
        # super(ModernTCNModel, self).__init__()
        super().__init__(**kwargs)
        self.network = ModernTCN(
            M=self.hparams.enc_in,
            L=self.hparams.input_size,
            T=self.hparams.output_size,
            D=self.hparams.hidden_size,
            P=P,
            S=S,
            kernel_sizes=kernel_sizes, 
            r=r, 
            num_layers=num_layers,
        )

    # 修改，锂电池预测
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_enc = x["encoder_cont"][:,:,:-1]
        # 输出
        prediction = self.network(x_enc)
        # 输出rescale， rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # 返回一个字典，包含输出结果（prediction）
        return self.to_network_output(prediction=prediction)


if __name__=='__main__':
    model=ModernTCN(11, L=8, D=96)
    input=torch.ones((100, 8, 11))
    out=model(input)
    print(out.shape)
