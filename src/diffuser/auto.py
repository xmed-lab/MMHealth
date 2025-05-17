# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : auto.py
# Time       ：11/11/2024 10:10 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：auto encoder; 如何在latent领域进行diffusion
"""
import faiss
import torch
import math
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from typing import List, Dict, Union, Callable, Optional
from pyhealth.tokenizer import Tokenizer


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class FrozenAutoencoderKL(nn.Module):
    # 冻结的特征编码器，可能对text不需要。
    def __init__(self, ddconfig, embed_dim, pretrained_path, scale_factor=0.18215):
        super().__init__()
        print(f'Create autoencoder with scale_factor={scale_factor}')
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        m, u = self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        assert len(m) == 0 and len(u) == 0
        self.eval()
        self.requires_grad_(False)

    def encode_moments(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def sample(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        return z

    def encode(self, x):
        moments = self.encode_moments(x) # 一个全局的特征，一个局部的特征生成z
        z = self.sample(moments)
        return z

    def decode(self, z):
        z = (1. / self.scale_factor) * z
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, inputs, fn):
        if fn == 'encode_moments':
            return self.encode_moments(inputs)
        elif fn == 'encode':
            return self.encode(inputs)
        elif fn == 'decode':
            return self.decode(inputs)
        else:
            raise NotImplementedError


def get_model(pretrained_path, scale_factor=0.18215):
    ddconfig = dict(
        double_z=True,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0
    )
    return FrozenAutoencoderKL(ddconfig, 4, pretrained_path, scale_factor)




########### 下面是离散的部分, 潜变量的采样和解码，估计得靠grounding top avg来作为输出了


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attn_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        ############################################################
        attn_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if attn_mask is not None:
            attn_mask = attn_mask.view_as(attn_score)
            if attn_mask.dtype == torch.bool:
                attn_score.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_score += attn_mask
        ############################################################

        A = torch.softmax(attn_score, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, input):
        X, attn_mask = input
        return self.mab(X, X, attn_mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.FloatTensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.num_inds = num_inds

    def forward(self, input):
        X, attn_mask = input
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attn_mask) # [B*V, num_inds, dim]
        attn_mask = attn_mask.transpose(-2, -1)
        return self.mab1(X, H, attn_mask)
    # def forward(self, X):
    #     H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
    #     return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim)) # [1, K, dim]
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X) #[batch_size*visit_len, K, dim]


class VisitEncode(nn.Module):
    def __init__(self, visit_config, tokenizers):
        super().__init__()
        self.config = visit_config
        self.mode = self.config['mode']
        self.embeddings = self.get_embedding_layers(tokenizers, self.config['dim'])
        self.set_encoder = ISAB(self.config['dim'], self.config['dim'], self.config['num_heads'], self.config['num_ids'], ln=False)

    @staticmethod
    def get_embedding_layers(
        feature_tokenizers: Dict[str, Tokenizer],
        embedding_dim: int,
    ) -> nn.ModuleDict:
        """Gets the default embedding layers using the feature tokenizers.

        These function is used for specific healthcare models, such as gamenet, safedrug, etc.

        Args:
            feature_tokenizers: a dictionary of feature tokenizers with keys
                corresponding to `self.feature_keys`.
            embedding_dim: the dimension of the embedding.

        Returns:
            embedding_layers: a module dictionary of embedding layers with keys
                corresponding to `self.feature_keys`.
        """
        embedding_layers = nn.ModuleDict()
        for key, tokenizer in feature_tokenizers.items():
            embedding_layers[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )
        return embedding_layers

    def set_encoder(self, input, attn_mask, rep=2):
        if self.config['mode']=='isab':
            attn_mask = attn_mask.unsqueeze(2).repeat(1,1,self.num_inds, 1)
            for i in range(rep):
                input = self.isab([input, attn_mask])
        elif self.config['mode'] =='sum':
            input = input.sum(dim=1, keepdim=True)
        return input


    def forward(self, batch_visit, batch_mask, feature_key, aggregate=True):
        """
        :param batch_visit: B, V, M
        :param batch_mask:  B,V,M
        :return: B, T, D
        """
        batch_visit = self.embeddings[feature_key](batch_visit)
        b, v, m, d = batch_visit.shape

        if self.mode == 'attn':
            batch_mask = batch_mask.float().masked_fill(batch_mask == 0, -1e9).masked_fill(batch_mask == 1,
                                                                                           float(0.0))  # 不大一样
            batch_visit = batch_visit.view(b * v, m, d)
            masks = batch_mask.view(b * v, m).unsqueeze(1).repeat(1, self.nhead, 1)
            batch_visit = self.set_encoder(batch_visit, masks)
            batch_visit = batch_visit.view(b, v, m, d)
        if aggregate:
            batch_visit = batch_visit.sum(dim=2) # B, 1,M,D-> B,1,D
        return batch_visit

    def decode(self):
        # 使用近似的样本表征作为sample到的潜变量
        pass




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ClusterEncode(nn.Module):
    def __init__(self, cluster_config):
        super().__init__()
        self.config = cluster_config
        self.mode = self.config['mode']
        self.k = self.config['k']
        self.proto_reg = self.config['proto_reg']
        self.ssl_temp = self.config['ssl_temp']
        self.bn = nn.BatchNorm1d(self.config['dim'], affine=False)
        self.register_buffer("visit_centroids", torch.randn(self.k, 3 * self.config['dim'])) # 这里的3就是feature_key
        nn.init.xavier_uniform_(self.visit_centroids)


    #
    # def forward_knn(self, x_train, x_query, k, device):
    #     """
    #     Perform KNN search using FAISS.
    #
    #     Parameters:
    #     - x_train: Training data (torch.Tensor).
    #     - x_query: Query data (torch.Tensor).
    #     - k: Number of nearest neighbors to retrieve.
    #     - device: Device to use for PyTorch tensors.
    #
    #     Returns:
    #     - indices: Indices of the nearest neighbors (torch.Tensor).
    #     - distances: Distances to the nearest neighbors (torch.Tensor).
    #     """
    #     # Convert training data to NumPy
    #     x_train_np = x_train.cpu().numpy().astype('float32')
    #
    #     # Create FAISS index
    #     index = faiss.IndexFlatL2(x_train.shape[1])  # 使用 L2 距离
    #     index.add(x_train_np)  # 添加训练数据到索引
    #
    #     # Convert query data to NumPy
    #     x_query_np = x_query.cpu().numpy().astype('float32')
    #
    #     # Search for k nearest neighbors
    #     distances, indices = index.search(x_query_np, k)
    #
    #     # Convert results to PyTorch tensors and move to specified device
    #     indices_tensor = torch.LongTensor(indices).to(device)
    #     distances_tensor = torch.FloatTensor(distances).to(device)
    #
    #     return indices_tensor, distances_tensor
    #
    # def nce_loss_knns(self, x, x_pair):
    #     norm_x_emb1 = x
    #     norm_x_pair_emb2 = x_pair
    #     c = self.bn(norm_x_emb1).T @ self.bn(norm_x_pair_emb2)  # [D, D]希望它更接近于单位对角矩阵
    #
    #     # sum the cross-correlation matrix between all gpus
    #     c.div_(self.args.batch_size)
    #     torch.distributed.all_reduce(c)
    #
    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # 对角线上的损失
    #     off_diag = off_diagonal(c).pow_(2).sum()  # 非对角线损失
    #     ssl_loss = on_diag + self.args.lambd * off_diag
    #     return ssl_loss

    def nce_loss_kmeans(self, x, centroids):
        """
        :param x: sample emb
        :param index: index, x在整体的里面的index。
        :return:
        """
        b, v , _ = x.shape
        norm_x_embeddings = F.normalize(x.view(b,-1)) # B,3D
        x2centroids = centroids # B, 3D
        pos_score_x = torch.mul(norm_x_embeddings, x2centroids).sum(dim=1)
        pos_score_x = torch.exp(pos_score_x / self.ssl_temp)
        ttl_score_x = torch.matmul(norm_x_embeddings, self.visit_centroids.transpose(0, 1))
        ttl_score_x = torch.exp(ttl_score_x / self.ssl_temp).sum(dim=1)
        proto_nce_loss_x = -torch.log(pos_score_x / ttl_score_x).sum()

        proto_nce_loss = self.proto_reg * proto_nce_loss_x

        return proto_nce_loss



    def forward(self, x, centroids):
        """
        :param x: batch_emb B * D
        :param centroids: cluster centroids B * D
        :return:
        """
        if self.mode == 'k-means':
            b,v,d = x.shape
            loss = self.nce_loss_kmeans(x, centroids)
            aug_samples = centroids.view(b,v,d) # 也可以用linear层
        else:
            raise NotImplementedError
        # elif self.mode == 'knn':
        #     indices, distances = self.forward_knn(search_sample, x, self.k, self.device)
        #     loss = self.nce_loss_knns(search_sample, x[indices])
        #     aug_samples = x[indices] # [B, 2D]

        return loss, aug_samples


class HistoryEncode(nn.Module):
    def __init__(self, history_config, feature_keys):
        super().__init__()
        self.config = history_config
        self.feature_keys = feature_keys  # 3个特征

        if self.config['mode'] == 'joint':
            self.config['dim'] = len(self.feature_keys) * self.config['dim']
        self.attention_dict = nn.ModuleDict({
            x: nn.MultiheadAttention(self.config['dim'], self.config['num_heads'], batch_first=True)
            for x in feature_keys + ['note']
        })
        # self.attention_dict['note'] = nn.TransformerEncoderLayer(batch_first=True, d_model=self.config['dim'], nhead=self.config['num_heads'])
        self.attention_joint = nn.MultiheadAttention(self.config['dim'], self.config['num_heads'], batch_first=True)
        self.clz_token = nn.Parameter(torch.randn(1, self.config['dim']))

    def add_clz_mask(self, mask):
        # 在掩码前添加 CLZ token
        batch_size, seq_length = mask.shape
        clz_mask = torch.ones(batch_size, 1).to(mask.device)  # CLZ token 的掩码为 1
        return torch.cat((clz_mask, mask), dim=1).bool()

    def add_clz_token(self, batch_data, clz_token):
        # 在每个句子前添加 CLZ token
        batch_size, seq_length, embed_size = batch_data.shape # B,V,D
        # 创建 CLZ token的形状为 (B, 1, D)
        clz_token = clz_token.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 CLZ token 和原始数据
        return torch.cat((clz_token, batch_data), dim=1)

    def forward(self, **kwargs):
        """可能会出现Nan值"""
        b,v,d = kwargs['conditions'].shape
        # individual attention
        if self.config['hist_mode'] == 'individual':
            tmp = []
            for feature_key in self.feature_keys:
                mask = kwargs[feature_key + '_hist_mask'].sum(dim=-1) !=0 # B,V,M-> B,V+1
                mask = self.add_clz_mask(mask)
                x = self.add_clz_token(kwargs[feature_key + '_hist'], self.clz_token) # B,V+1,D
                batch_history, _ = self.attention_dict[feature_key](x, x, x, key_padding_mask=~mask) # B,V+1,D
                tmp.append(batch_history[:,0,:]) # B, D
            if kwargs['has_note']:
                mask = kwargs['note'+'_hist'].sum(dim=-1)!=0
                mask = self.add_clz_mask(mask)
                x = self.add_clz_token(kwargs['note'+'_hist'], self.clz_token)
                batch_history,_= self.attention_dict['note'](x, x, x, key_padding_mask=~mask) # attention挺难的。
                # batch_history= self.attention_dict['note'](x, src_key_padding_mask=~mask) # attention挺难的。
                tmp.append(batch_history[:,0,:]) # B, D
            tmp = torch.stack(tmp, dim=1) # B, 3, D
        elif self.config['hist_mode'] == 'joint':
            tmp = [kwargs[feature_key + '_hist'] for feature_key in self.feature_keys]
            tmp = torch.cat(tmp, dim=-1) # B, K, 3D
            tmp_mask = [kwargs[feature_key + '_hist_mask'].sum(dim=-1)!=0 for feature_key in self.feature_keys] # B,K
            # if kwargs['has_note']:
            #     tmp = torch.cat([tmp, kwargs['note'+'_hist']], dim=-1)
            #     tmp_mask.append(kwargs['note'+'_hist'].sum(dim=-1)!=0)

            tmp = self.add_clz_token(tmp, self.clz_token)
            tmp_mask = torch.stack(tmp_mask, dim=1) # B,3,K
            tmp_mask = self.add_clz_mask(tmp_mask.sum(dim=1)!=0)
            tmp = self.attention_joint(tmp, tmp, tmp, key_padding_mask=~tmp_mask)
            tmp = tmp[:,0,:].view(b, len(self.feature_keys), d) # B,3,D
            if kwargs['has_note']:
                mask = kwargs['note'+'_hist'].sum(dim=-1)!=0
                mask = self.add_clz_mask(mask)
                x = self.add_clz_token(kwargs['note'+'_hist'], self.clz_token)
                # batch_history= self.attention_dict['note'](x, src_key_padding_mask=~mask) # attention挺难的。
                # tmp.append(batch_history[:,0,:]) # B, D
                batch_history,_= self.attention_dict['note'](x, x, x, key_padding_mask=~mask) # attention挺难的。
                tmp = torch.cat([tmp, batch_history[:,0,:].unsqueeze(1)], dim=1) # B, 4, D
        return tmp






class MaskEncode(nn.Module):
    def __init__(self, mask_config):
        super().__init__()
        self.config = mask_config
        self.mask_emb = nn.Embedding(2, self.config['dim'])

    def forward(self, batch_mask):
        mask_emb = self.mask_emb(batch_mask)
        return mask_emb




class ContextEncoder(nn.Module):
    def __init__(self, con_config, cluster_config, feature_keys):
        super().__init__()
        self.config = con_config
        self.feature_keys = feature_keys
        self.mask_encoder = MaskEncode(con_config)
        self.history_encoder = HistoryEncode(con_config, feature_keys)
        self.cluster = ClusterEncode(cluster_config) # embed在forward处传入。
        self.mode = con_config['mode']

    def get_context(self, mask_emb, history_emb, neigh_emb):
        # 获取对应的context emb
        # neigh_emb = torch.zeros_like(neigh_emb) # aba
        if self.mode == 'concat':
            context_emb = torch.cat([mask_emb, history_emb, neigh_emb], dim=1) # B, 9 （3+3+3+1）, D
            return context_emb
        else:
            raise NotImplementedError


    def forward(self, x, batch_centroids_emb,  **kwargs):
        mask_emb = self.mask_encoder(kwargs['mask'].long()) # B, 3, D
        history_emb = self.history_encoder(**kwargs) # B, 3, D （B，4，D）
        neigh_loss, neigh_emb = self.cluster(x, batch_centroids_emb) # B,3,D 和B, 3, D

        context_emb = self.get_context(mask_emb, history_emb, neigh_emb)

        return neigh_loss, context_emb




