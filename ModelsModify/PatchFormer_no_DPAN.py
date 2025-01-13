from pytorch_forecasting.models import BaseModel
from typing import Dict
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from torch.nn import init

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

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


#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=4):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

class FeatureAttentionSE(nn.Module):
    '''
    基于SE原理，首先将不关注的维度进行自适应pooling。
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Conv_Squeeze = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        # [n, c, len] -> [n, c, 1]
        z = self.avgpool(U)            
        # [n, c, 1]-> [n, c/2, 1]
        z = self.Conv_Squeeze(z)      
        # [n, c/2, 1]->[n, c, 1]
        z = self.Conv_Excitation(z)    
        # [n, c, 1]
        z = self.sigmoid(z)
        # [n, c,len]*[n, c,len] ,其中：expand_as，将c复制到所有维度 [n, c,len]
        outputs=U * z.expand_as(U)
        return outputs
    
class FeatureAttentionSE4D(nn.Module):
    '''
    基于SE原理，首先将不关注的维度进行自适应pooling。
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1,1), bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=(1,1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, U):
        # 输入：[100,8,48,100]
        # [n, c, len, feas] -> [n, c, 1, 1]       [100, 8, 48, 100] -> [100, 8, 1, 1]
        z = self.avgpool(U)             # shape: [bs, c, h, w] to [bs, c, 1, 1]
        # [n, c, 1, 1] -> [n, c/2, 1, 1]
        z = self.Conv_Squeeze(z)        # shape: [bs, c/2, 1, 1]
        # [n, c/2, 1, 1] -> [n, c, 1, 1]
        z = self.Conv_Excitation(z)     # shape: [bs, c, 1, 1]
        # [n, c, 1, 1]
        z = self.sigmoid(z)
        # 归一化  [n, c, 1, 1]
        z= self.softmax(z)
        # [n, c,len]*[n, c,len] ,其中：expand_as，将c复制到所有维度 [n, c,len]
        outputs = U * z.expand_as(U)
        # outputs:(100,8,48,100)    z:(100,8,1,1)
        return outputs

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key_value):
        B, N_q, C = query.shape  # B: batch size, N_q: number of query tokens, C: Embedding dimension
        B, N_kv, C = key_value.shape  # B: batch size, N_kv: number of key-value tokens, C: Embedding dimension

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_q, head_dim)
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, N_kv, head_dim)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.dropout_layer(x)
        return x

class IntraPatchCrossAttention(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, dropout=0.0):
        super(IntraPatchCrossAttention, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multi_head_cross_attention = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

    def forward(self, x, global_context):
        B, L, C = x.shape  # B: batch size, L: sequence length, C: Embedding dimension
        assert L % self.patch_size == 0, "Sequence length must be divisible by patch size"
        num_patches = L // self.patch_size

        # Reshape input to (B, num_patches, patch_size, C)
        x = x.reshape(B, num_patches, self.patch_size, C)

        # Apply intra-patch cross-attention to each patch
        x = x.permute(0, 1, 3, 2)  # (B, num_patches, C, patch_size)
        x = x.reshape(B * num_patches, C, self.patch_size).permute(0, 2, 1)  # (B * num_patches, patch_size, C)
        x = self.multi_head_cross_attention(x, global_context)  # (B * num_patches, patch_size, C)
        x = x.reshape(B, num_patches, self.patch_size, C).permute(0, 1, 3, 2)  # (B, num_patches, C, patch_size)

        # Reshape back to original shape
        x = x.reshape(B, L, C)
        return x
    
class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))



        attention = torch.matmul(query, key)
        attention /= (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x, attention

class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class VarEncoder(nn.Module):
    def __init__(self, n_var, dim):
        super().__init__()
        self.n_var = n_var
        self.dim = dim
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(1, dim))
                                     for _ in range(n_var)])

    def forward(self, x):
        # [B, L, C]
        y = torch.zeros([x.shape[0], x.shape[1], self.n_var, self.dim], device=x.device)
        for i in range(self.n_var):
            y[:, :, i, :] = self.layers[i](x[:, :, i].unsqueeze(-1))
        return y


class PatchFormer_no_DPAN(nn.Module):
    def __init__(self,patch_len,seq_len,pred_len,
                 enc_in, d_model=16,factor=3, dropout=0.1, output_attention=False, n_heads=8, activation='gelu', e_layers=2):
        super(PatchFormer_no_DPAN,self).__init__()

        self.patch_nums = seq_len // patch_len  # patch_len=stride
        self.patch_size = patch_len
        self.encoder_d_model = seq_len      # d_model of encoder
        self.d_ff = 2*self.encoder_d_model

        self.var_embedding = VarEncoder(enc_in, d_model)
        # ##intra_patch_attention
        # self.intra_embeddings = nn.Parameter(torch.rand(self.patch_nums, 1, 1, enc_in, 16),
        #                                         requires_grad=True)
        # self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
        #     nn.Linear(16, d_model)]) for _ in range(self.patch_nums)])
        # self.intra_d_model = d_model
        # self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=True)
        # self.weights_generator_distinct = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=16, num_nodes=enc_in,
        #                                                   factorized=True, number_of_weights=2)
        # self.weights_generator_shared = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=None, num_nodes=enc_in,
        #                                                 factorized=False, number_of_weights=2)
        # self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums*self.patch_size)

        # ##inter_patch_attention
        # self.stride = self.patch_size
        # self.inter_patch_attention = FeatureAttentionSE4D(self.patch_nums)


        # ##temporal_feature_fusion
        # self.fusion = FeatureAttentionSE(2*seq_len)

        ##feature_attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), self.encoder_d_model, n_heads),
                    self.encoder_d_model,
                    self.d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.encoder_d_model)
        )   
        self.head = nn.Linear(seq_len*d_model,pred_len)

    def forward(self,x):
        batch_size,L,M = x.shape 
        intra_out_concat = None

        # weights_shared, biases_shared = self.weights_generator_shared()
        # weights_distinct, biases_distinct = self.weights_generator_distinct()

        new_x = self.var_embedding(x)#[B,L,M]->[B,L,M,D]       
        # ####Intra-patch Attention##### 
        # for i in range(self.patch_nums):
        #     t = new_x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :]

        #     intra_emb = self.embeddings_generator[i](self.intra_embeddings[i]).expand(batch_size, -1, -1,  -1)
        #     t = torch.cat([intra_emb, t], dim=1)
        #     out, attention = self.intra_patch_attention(intra_emb, t, t, weights_distinct, biases_distinct, weights_shared,
        #                                        biases_shared)

        #     if intra_out_concat == None:
        #         intra_out_concat = out

        #     else:
        #         intra_out_concat = torch.cat([intra_out_concat, out], dim=1) # [B,N,M,D]      
        # intra_out_concat = intra_out_concat.permute(0,3,2,1)
        # intra_out_concat = self.intra_Linear(intra_out_concat)
        # intra_out_concat = intra_out_concat.permute(0,3,2,1)
        # intra_out_concat += new_x #[B,L,M,D]

        # ####Inter-patch Attention#####
        # x = new_x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # [b x patch_num x nvar x dim x patch_len]
        # x = x.permute(0, 2, 1, 3, 4)  # [b x nvar x patch_num x dim x patch_len ]
        # b, nvar, patch_num, dim, patch_len = x.shape
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[-1]))  # [b*nvar, patch_num, dim, patch_len]        
        # inter_out = self.inter_patch_attention(x) # [B*M, N, D, P] -> [B*M, N, D, P]
        # inter_out = torch.reshape(inter_out, (b, nvar, patch_num, self.patch_size, dim))
        # inter_out = torch.reshape(inter_out, (b, self.patch_size * self.patch_nums, nvar, dim))  # [b, temporal, nvar, dim]
        # inter_out += new_x #[B,L,M,D]

        ####temporal FeatureFusion#####
        # out = torch.concat([intra_out_concat, inter_out],dim=1)#2*[B,L,M,D]->[B,2*L,M,D]
        # out = torch.sum(out, dim=-2)#[B,2*L,M,D]->[B,2*L,D]
        # out = self.fusion(out)#[B,2*L,D]->[B,2*L,D]
        out = new_x
        out = torch.sum(out, dim=-2)
        out = rearrange(out,'b l d -> b d l')  #[B,2*L,D]->[B,D,2*L]

        ####Feature Attention#####
        out,_ = self.encoder(out)   #[B,D,2*L]->[B,D,2*L]

        ####Projection#####        
        z = rearrange(out, 'b d l -> b (d l)')  # [B, D, 2*L] -> [B, 2*D*L]
        z = self.head(z)  # [B, 2*D*L] -> [B, T]

        return z


class PatchFormer_no_DPANNetModel(BaseModel):
    def __init__(self,patch_len,seq_len,pred_len,enc_in, 
                 d_model=16,factor=3, dropout=0.1, output_attention=False, n_heads=4, activation='gelu', e_layers=1,**kwargs):

        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        # super().__init__()
        # super(ModernTCNModel, self).__init__()
        super().__init__(**kwargs)
        self.network = PatchFormer_no_DPAN(
            patch_len=self.hparams.patch_len,
            enc_in=self.hparams.enc_in,
            seq_len=self.hparams.seq_len,
            pred_len=self.hparams.pred_len,
            d_model=self.hparams.d_model,
            factor=self.hparams.factor, 
            dropout=self.hparams.dropout,
            output_attention=self.hparams.output_attention, 
            n_heads=self.hparams.n_heads, 
            activation=self.hparams.activation, 
            e_layers=self.hparams.e_layers
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
    mymodel = PatchFormer_no_DPAN(
            patch_len=2,
            enc_in=1,
            seq_len=32,
            pred_len=1)
    # input_size, output_size, d_model, enc_in, factor, dropout, output_attention, n_heads, d_ff, activation, e_layers
    src = torch.rand((128, 32, 1))
    out = mymodel(src)
    print(out.shape)