import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from math import sqrt
import numpy as np
from pytorch_forecasting.models import BaseModel
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): 
        x = x.permute(0, 2, 1) 
        if x_mark is None:
            x = self.value_embedding(x) 
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.mark_embedding = nn.Linear(d_inp, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): # [B, L, M] and [B, L, 4] -> [B, L, d]

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.mark_embedding(x_mark)
        # x: [Batch Variate d_model]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


# factor:probsparse attn factor
class Backbone1(nn.Module):
    def __init__(self, input_size, output_size, d_model, enc_in, factor, dropout, output_attention, n_heads, d_ff, activation, e_layers):
        super(Backbone1, self).__init__()
        # Embedding
        self.embed = DataEmbedding_inverted(input_size, d_model)
        self.flatten = nn.Flatten()
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.predict = nn.Linear(d_model, output_size)

    def forward(self, x): # [B, L, M] and [B, L, 4] -> [B, H]
        B, _, D = x.shape

        x_embed = self.embed(x, None)
        y, _ = self.encoder(x_embed)
        y = y[:, 0, :] 
        y = self.predict(y) 

        return y # [B, H]




class iTransformer(nn.Module):

    def __init__(self, input_size, output_size, d_model, enc_in, factor, dropout, output_attention, n_heads, d_ff, activation, e_layers):
        super(iTransformer, self).__init__()

        self.backbone = Backbone1(input_size, output_size, d_model, enc_in, factor, dropout, output_attention, n_heads, d_ff, activation, e_layers)

        # self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len

    def forward(self, x):
        y = self.backbone(x) # [B, L, M] and [B, L, 4] -> [B, H]
        return y


class iTransformerModel(BaseModel):
    def __init__(self,input_size=24, output_size=1, d_model=16, enc_in=7, factor=3, dropout=0.1, output_attention=False, 
                n_heads=8, d_ff=32, activation='gelu', e_layers=2,**kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        # super().__init__()
        # super(ModernTCNModel, self).__init__()
        super().__init__(**kwargs)
        self.network = iTransformer(
            input_size=input_size,
            output_size=output_size,
            d_model=d_model,
            enc_in=enc_in,
            factor=factor,
            dropout=dropout,
            output_attention=output_attention,
            n_heads=n_heads,
            d_ff=d_ff,
            activation=activation,
            e_layers=e_layers,
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
    # 源代码的iTransformer的输入是(B, L, D)
    # iTransformer里面没有factor这个参数,只有Informer里面有这个参数
    # input_size = 104
    # output_size=16
    # d_model=128
    # enc_in=4  # num_input
    # factor =5
    # dropout=0.01
    # output_attention=False
    # n_heads=4
    # d_ff=256
    # activation='relu'
    # e_layers=1
    # mymodel = iTransformer(104,16,128,4,5,0.01,False,4,64,'relu',1)
    # # input_size, output_size, d_model, enc_in, factor, dropout, output_attention, n_heads, d_ff, activation, e_layers
    # src = torch.rand((128, 104, 4))
    # out = mymodel(src)
    # print(out.shape)
    # torch.Size([128, 16])
    input_size = 16
    output_size=1
    d_model=128
    enc_in=1  # num_input
    factor =5
    dropout=0.1
    output_attention=False
    n_heads=8
    d_ff=256
    activation='relu'
    e_layers=1
    mymodel = iTransformer(104,16,128,4,5,0.01,False,4,64,'relu',1)
    # input_size, output_size, d_model, enc_in, factor, dropout, output_attention, n_heads, d_ff, activation, e_layers
    src = torch.rand((128, 104, 4))
    out = mymodel(src)
    print(out.shape)









