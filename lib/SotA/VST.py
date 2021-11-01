

import math
import torch
import numpy as np

from torch import nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models import load_checkpoint

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(dim, dim)

        self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb_fea, depth_fea):
        B, N, C = rgb_fea.shape

        rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # rgb branch
        rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.proj_drop(rgb_fea)

        # depth branch
        depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.proj_drop(depth_fea)

        return rgb_fea, depth_fea


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MutualSelfBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # mutual attention
        self.norm1_rgb_ma = norm_layer(dim)
        self.norm2_depth_ma = norm_layer(dim)
        self.mutualAttn = MutualAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3_rgb_ma = norm_layer(dim)
        self.norm4_depth_ma = norm_layer(dim)
        self.mlp_rgb_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # rgb self attention
        self.norm1_rgb_sa = norm_layer(dim)
        self.selfAttn_rgb = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_rgb_sa = norm_layer(dim)
        self.mlp_rgb_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # depth self attention
        self.norm1_depth_sa = norm_layer(dim)
        self.selfAttn_depth = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_depth_sa = norm_layer(dim)
        self.mlp_depth_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, rgb_fea, depth_fea):

        # mutual attention
        rgb_fea_fuse, depth_fea_fuse = self.drop_path(self.mutualAttn(self.norm1_rgb_ma(rgb_fea), self.norm2_depth_ma(depth_fea)))

        rgb_fea = rgb_fea + rgb_fea_fuse
        depth_fea = depth_fea + depth_fea_fuse

        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_ma(self.norm3_rgb_ma(rgb_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_ma(self.norm4_depth_ma(depth_fea)))

        # rgb self attention
        rgb_fea = rgb_fea + self.drop_path(self.selfAttn_rgb(self.norm1_rgb_sa(rgb_fea)))
        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_sa(self.norm2_rgb_sa(rgb_fea)))

        # depth self attention
        depth_fea = depth_fea + self.drop_path(self.selfAttn_depth(self.norm1_depth_sa(depth_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_sa(self.norm2_depth_sa(depth_fea)))

        return rgb_fea, depth_fea


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.rgb_norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_fea):

        for block in self.blocks:
            rgb_fea = block(rgb_fea)

        rgb_fea = self.rgb_norm(rgb_fea)

        return rgb_fea


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        return fea

class Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(Transformer, self).__init__()

        self.encoderlayer = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):

        rgb_memory = self.encoderlayer(rgb_fea)

        return rgb_memory


class saliency_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, 1:-1, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-2, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-1, :]
        return infer_fea


class contour_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, -1, :].unsqueeze(1), x[:, 1:-1, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attn = attn.softmax(dim=-1)
        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-2, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-1, :]
        return infer_fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.saliency_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.contour_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)

    def forward(self, rgb_fea):
        B, _, _ = rgb_fea.shape
        fea_1_16 = self.mlp_s(self.norm(rgb_fea))   # [B, 14*14, 384]

        saliency_tokens = self.saliency_token.expand(B, -1, -1)
        fea_1_16 = torch.cat((saliency_tokens, fea_1_16), dim=1)

        contour_tokens = self.contour_token.expand(B, -1, -1)
        fea_1_16 = torch.cat((fea_1_16, contour_tokens), dim=1)
        # fea_1_16 [B, 1 + 14*14 + 1, 384]

        fea_1_16 = self.encoderlayer(fea_1_16)
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        saliency_tokens = fea_1_16[:, 0, :].unsqueeze(1)
        contour_tokens = fea_1_16[:, -1, :].unsqueeze(1)

        saliency_fea_1_16 = self.saliency_token_pre(fea_1_16)
        contour_fea_1_16 = self.contour_token_pre(fea_1_16)
        return saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class crosstask_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        # self.kqv = nn.Linear(dim, 3 * self.emb)
        self.q_s = nn.Linear(dim, self.emb)
        self.k_s = nn.Linear(dim, self.emb)
        self.v_s = nn.Linear(dim, self.emb)

        self.q_c = nn.Linear(dim, self.emb)
        self.k_c = nn.Linear(dim, self.emb)
        self.v_c = nn.Linear(dim, self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj_s = nn.Linear(self.emb, self.emb)
        self.proj_c = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1_s = nn.LayerNorm(dim)
        self.norm1_c = nn.LayerNorm(dim)
        self.norm2_s = nn.LayerNorm(self.emb)
        self.norm2_c = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp_s = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m_s = int(self.emb * kernel_ratio)
        self.w_s = torch.randn(self.m_s, self.emb)
        self.w_s = nn.Parameter(nn.init.orthogonal_(self.w_s) * math.sqrt(self.m_s), requires_grad=False)

        self.m_c = int(self.emb * kernel_ratio)
        self.w_c = torch.randn(self.m_c, self.emb)
        self.w_c = nn.Parameter(nn.init.orthogonal_(self.w_c) * math.sqrt(self.m_c), requires_grad=False)

    def prm_exp_s(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m_s) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w_s)

        return torch.exp(wtx - xd) / math.sqrt(self.m_s)

    def prm_exp_c(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m_c) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w_c)

        return torch.exp(wtx - xd) / math.sqrt(self.m_c)

    def cross_attn(self, saliency_fea, contour_fea):
        k_s, q_s, v_s = self.k_s(saliency_fea), self.q_s(saliency_fea), self.v_s(saliency_fea)
        k_c, q_c, v_c = self.k_c(contour_fea), self.q_c(contour_fea), self.v_c(contour_fea)

        kp_s, qp_s = self.prm_exp_s(k_c), self.prm_exp_s(q_s)  # (B, T, m), (B, T, m)
        D_s = torch.einsum('bti,bi->bt', qp_s, kp_s.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv_s = torch.einsum('bin,bim->bnm', v_c.float(), kp_s)  # (B, emb, m)
        y_s = torch.einsum('bti,bni->btn', qp_s, kptv_s) / (D_s.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y_s = saliency_fea + self.dp(self.proj_s(y_s))  # same as token_transformer in T2T layer, use v as skip connection
        y_s = self.dp(self.proj_s(y_s))  # same as token_transformer in T2T layer, use v as skip connection

        kp_c, qp_c = self.prm_exp_c(k_s), self.prm_exp_c(q_c)  # (B, T, m), (B, T, m)
        D_c = torch.einsum('bti,bi->bt', qp_c, kp_c.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv_c = torch.einsum('bin,bim->bnm', v_s.float(), kp_c)  # (B, emb, m)
        y_c = torch.einsum('bti,bni->btn', qp_c, kptv_c) / (D_c.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y_c = contour_fea + self.dp(self.proj_c(y_c))  # same as token_transformer in T2T layer, use v as skip connection
        y_c = self.dp(self.proj_c(y_c))  # same as token_transformer in T2T layer, use v as skip connection

        return y_s, y_c

    def forward(self, saliency_fea, contour_fea):
        # cross task attention
        saliency_fea_fuse, contour_fea_fuse = self.cross_attn(self.norm1_s(saliency_fea), self.norm1_c(contour_fea))

        saliency_fea = saliency_fea + saliency_fea_fuse
        contour_fea = contour_fea + contour_fea_fuse

        saliency_fea = saliency_fea + self.mlp_s(self.norm2_s(saliency_fea))
        contour_fea = contour_fea + self.mlp_c(self.norm2_c(contour_fea))

        return saliency_fea, contour_fea


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # x [B, 56*56, 147=7*7*3]
        # iteration1: restricturization/reconstruction
        x_1_4 = self.attention1(x)
        B, new_HW, C = x_1_4.shape
        x = x_1_4.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: restricturization/reconstruction
        x_1_8 = self.attention2(x)
        B, new_HW, C = x_1_8.shape
        x = x_1_8.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x, x_1_8, x_1_4


class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, x_1_8, x_1_4 = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # T2T-ViT backbone
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # return x[:, 0]
        return x[:, 1:, :], x_1_8, x_1_4

    def forward(self, x):
        x, x_1_8, x_1_4 = self.forward_features(x)
        # x = self.head(x)
        return x, x_1_8, x_1_4



def T2t_vit_t_14(pretrained=False):  # adopt transformers for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 384 ** -0.5)

    # model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_checkpoint(model, 'data/backbone_ckpt/80.7_T2T_ViT_t_14.pth.tar', use_ema=True)
        print('Model loaded from {}'.format('data/backbone_ckpt/80.7_T2T_ViT_t_14.pth.tar'))
    return model

class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens, contour_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = torch.cat((fea, contour_tokens), dim=1)
        # [B, 1 + H*W + 1, 384]

        fea = self.encoderlayer(fea)
        # fea [B, 1 + H*W + 1, 384]
        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        contour_tokens = fea[:, -1, :].unsqueeze(1)

        saliency_fea = self.saliency_token_pre(fea)
        # saliency_fea [B, H*W, 384]
        contour_fea = self.contour_token_pre(fea)
        # contour_fea [B, H*W, 384]

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        contour_fea = self.mlp2_c(self.norm2_c(contour_fea))

        return saliency_fea, contour_fea, fea, saliency_tokens, contour_tokens


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)

        return dec_fea


class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_c = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        # predict contour maps
        self.pre_1_16_c = nn.Linear(token_dim, 1)
        self.pre_1_8_c = nn.Linear(token_dim, 1)
        self.pre_1_4_c = nn.Linear(token_dim, 1)
        self.pre_1_1_c = nn.Linear(token_dim, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4):
        # saliency_fea_1_16 [B, 14*14, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain saliency token and contour token)

        # saliency_tokens [B, 1, 384]
        # contour_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 56*56, 64]

        B, _, _, = token_fea_1_16.size()

        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # saliency_fea_1_16 [B, 14*14, 64]
        mask_1_16 = self.pre_1_16(saliency_fea_1_16)
        mask_1_16 = mask_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        contour_fea_1_16 = self.mlp_c(self.norm_c(contour_fea_1_16))
        # contour_fea_1_16 [B, 14*14, 64]
        contour_1_16 = self.pre_1_16_c(contour_fea_1_16)
        contour_1_16 = contour_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        fea_1_8 = self.decoder1(token_fea_1_16[:, 1:-1, :], rgb_fea_1_8)

        # token prediction
        saliency_fea_1_8, contour_fea_1_8, token_fea_1_8, saliency_tokens, contour_tokens = self.token_pre_1_8(fea_1_8, saliency_tokens, contour_tokens)

        # predict saliency maps and contour maps
        mask_1_8 = self.pre_1_8(saliency_fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        contour_1_8 = self.pre_1_8_c(contour_fea_1_8)
        contour_1_8 = contour_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(token_fea_1_8[:, 1:-1, :], rgb_fea_1_4)

        # token prediction
        saliency_fea_1_4, contour_fea_1_4, token_fea_1_4, saliency_tokens, contour_tokens = self.token_pre_1_4(fea_1_4, saliency_tokens, contour_tokens)

        # predict saliency maps and contour maps
        mask_1_4 = self.pre_1_4(saliency_fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        contour_1_4 = self.pre_1_4_c(contour_fea_1_4)
        contour_1_4 = contour_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        saliency_fea_1_1 = self.decoder3(saliency_fea_1_4)
        contour_fea_1_1 = self.decoder3_c(contour_fea_1_4)

        mask_1_1 = self.pre_1_1(saliency_fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        contour_1_1 = self.pre_1_1_c(contour_fea_1_1)
        contour_1_1 = contour_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1], [contour_1_16, contour_1_8, contour_1_4, contour_1_1]

class VST(nn.Module):
    def __init__(self, depth, pretrained=False):
        super(VST, self).__init__()

        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=pretrained)

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=224)

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)

        return outputs[0][-1]