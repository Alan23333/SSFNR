import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
from timm.models.layers import DropPath
from einops import rearrange


def make_model(opt):
    return SSFNR(img_size=opt.patch_size,in_chans=opt.n_colors)


class FFT(nn.Module):
    def __init__(self, channels):
        super(FFT, self).__init__()
        self.channels = channels

        self.conv_amp = nn.Sequential(
            dep_conv(self.channels, kernel_size=1),
        )

        self.conv_pha = nn.Sequential(
            dep_conv(self.channels, kernel_size=1),
        )

    def forward(self, x):
        b, c, h, w  = x.size()

        fre = torch.fft.rfft2(x, norm='backward')

        amp = torch.abs(fre)
        pha = torch.angle(fre)

        amp_fea = self.conv_amp(amp)
        pha_fea = self.conv_pha(pha)

        real = amp_fea * torch.cos(pha_fea) + 1e-8
        imag = amp_fea * torch.sin(pha_fea) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(h, w), norm='backward'))

        return out


class Frequency_Domain_Learning(nn.Module):
    def __init__(self, in_channels):
        # bn_layer not used
        super(Frequency_Domain_Learning, self).__init__()

        self.conv_1 = nn.Sequential(
            depthwise_conv(in_channels=in_channels, out_channels=in_channels),
            nn.GELU()
        )

        self.fft = FFT(channels=in_channels)

        self.conv_2 = nn.Sequential(
            point_conv(in_channels, in_channels)
        )

    def forward(self, x, x_size):
        H, W = x_size
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.conv_1(x)

        output = self.fft(x)
        output = output + x
        output = self.conv_2(output)

        output = rearrange(output, "b c h w -> b (h w) c", h=H, w=W).contiguous()

        return output


def point_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1)


def dep_conv(in_channels, kernel_size):
    return nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2, groups=in_channels)


class depthwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(depthwise_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.point_conv = point_conv(self.in_channels, self.out_channels)
        self.depth_conv = dep_conv(in_channels=self.out_channels,
                                   kernel_size=self.kernel_size)

    def forward(self, x):
        out = self.point_conv(x)
        out = self.depth_conv(out)
        return out


class SE(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=32):
        super(SE, self).__init__()
        self.num_feat = num_feat
        self.cab = nn.Sequential(
            nn.Linear(num_feat, num_feat // compress_ratio),
            nn.GELU(),
            nn.Linear(num_feat // compress_ratio, num_feat))

        self.ca = ChannelAttention(num_feat, squeeze_factor, memory_blocks=128)

    def forward(self, x):
        y = self.cab(x)
        y = self.ca(y)
        x = x + y
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor),
        )
        self.m = nn.LeakyReLU
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            nn.Sigmoid())
        self.mb = torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        b, n, c = x.shape
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2
        return out


def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        hw = x.shape[1]
        h = int(math.sqrt(hw))
        x = self.fc1(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=h)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1)*x2 + self.act(x2)*x1
        x = rearrange(x, "b c h w -> b (h w) c", h=h, w=h)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Spatial_Domain_Learning(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=6, attn_drop=0., proj_drop=0.,
                 qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.tmp_H = H_sp
        self.tmp_W = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.resolution != H or self.resolution != W:
            if self.idx == -1:
                H_sp, W_sp = H, W
            elif self.idx == 0:
                H_sp, W_sp = H, self.split_size
            elif self.idx == 1:
                W_sp, H_sp = W, self.split_size
            else:
                print("ERROR MODE", self.idx)
                exit(0)
            self.H_sp = H_sp
            self.W_sp = W_sp
        else:
            self.H_sp = self.tmp_H
            self.W_sp = self.tmp_W

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        if self.position_bias:
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=attn.device)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=attn.device)
            biases = torch.stack(
                torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            coords_h = torch.arange(self.H_sp, device=attn.device)
            coords_w = torch.arange(self.W_sp, device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)

            pos = self.pos(biases)
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(-1, self.H_sp * self.W_sp, C)
        x = windows2img(x, self.H_sp, self.W_sp, H, W)
        return x


class SFB(nn.Module):
    def __init__(self, dim, reso, num_heads,
                 split_size=4, shift_size=0, mlp_ratio=4., s_scale=0.5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.s_scale = s_scale
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.channels = int(self.dim*self.s_scale)

        self.qkv = nn.Linear(self.channels, self.channels*3, bias=qkv_bias)

        assert 0 <= self.shift_size < self.split_size, "shift_size must in 0-split_size"

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.s_domain = nn.ModuleList([
            Spatial_Domain_Learning(
                dim=self.channels//2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads//2, dim_out=self.channels//2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])
        
        self.se = SE(num_feat=self.dim)

        mlp_hidden_dim = int(self.channels * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.locality = dep_conv(in_channels=self.channels, kernel_size=3)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.f_domain = Frequency_Domain_Learning(in_channels=dim-self.channels)

    def calculate_mask(self, H, W):
        img_mask_0 = torch.zeros((1, H, self.split_size, 1))
        img_mask_1 = torch.zeros((1, self.split_size, W, 1))
        slices = (slice(-self.split_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for s in slices:
            img_mask_0[:, :, s, :] = cnt
            img_mask_1[:, s, :, :] = cnt
            cnt += 1

        img_mask_0 = img_mask_0.view(1, H // H, H, self.split_size // self.split_size, self.split_size, 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, self.split_size, 1)
        mask_windows_0 = img_mask_0.view(-1, H * self.split_size)
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))
        num_v = W // self.split_size
        attn_mask_0_la = torch.zeros((num_v, H * self.split_size, H * self.split_size))
        attn_mask_0_la[-1] = attn_mask_0

        img_mask_1 = img_mask_1.view(1, self.split_size // self.split_size, self.split_size, W // W, W, 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size, W, 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size * W)
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        num_h = H // self.split_size
        attn_mask_1_la = torch.zeros((num_h, W * self.split_size, W * self.split_size))
        attn_mask_1_la[-1] = attn_mask_1

        return attn_mask_0_la, attn_mask_1_la

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)

        qkv = self.qkv(img[:,:,:self.channels])
        qkv = qkv.reshape(B, -1, 3, self.channels).permute(2, 0, 1, 3)

        v = qkv[2].transpose(-2, -1).contiguous().view(B, self.channels, H, W)

        if self.shift_size > 0:
            qkv = qkv.view(3, B, H, W, self.channels)
            qkv_0 = torch.roll(qkv[:, :, :, :, :self.channels//2], shifts=-self.shift_size, dims=3)
            qkv_0 = qkv_0.view(3, B, L, self.channels//2)

            qkv_1 = torch.roll(qkv[:, :, :, :, self.channels//2:self.channels], shifts=-self.shift_size, dims=2)
            qkv_1 = qkv_1.view(3, B, L, self.channels//2)

            if self.patches_resolution != H or self.patches_resolution != W:
                mask_tmp = self.calculate_mask(H, W)
                x1_shift = self.s_domain[0](qkv_0, H, W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.s_domain[1](qkv_1, H, W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.s_domain[0](qkv_0, H, W, mask=self.attn_mask_0)
                x2_shift = self.s_domain[1](qkv_1, H, W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=self.shift_size, dims=2)
            x2 = torch.roll(x2_shift, shifts=self.shift_size, dims=1)
            x1 = x1.view(B, L, self.channels//2).contiguous()
            x2 = x2.view(B, L, self.channels//2).contiguous()
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            x1 = self.s_domain[0](qkv[:, :, :, :self.channels//2], H, W).view(B, L, self.channels//2).contiguous()
            x2 = self.s_domain[1](qkv[:, :, :, self.channels//2: self.channels], H, W).view(B, L, self.channels//2).contiguous()
            attened_x = torch.cat([x1, x2], dim=2)

        lcm = self.locality(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, self.channels)
        attened_x = attened_x + lcm
        attened_f = self.f_domain(img[:, :, self.channels:], x_size)

        attened = torch.cat([attened_x, attened_f], dim=2)
        attened = rearrange(attened, 'b n (g d) -> b n (d g)', g=8)

        attened = attened + self.se(attened)

        attened = self.proj(attened)
        attened = self.proj_drop(attened)

        x = x + self.drop_path(attened)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=10, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class SSFNR(nn.Module):
    def __init__(self,
                 img_size=96,
                 in_chans=10,
                 dim=48,
                 depth=[4, 6, 6, 8],
                 split_size_0=[4, 4, 4, 4],
                 num_heads=[2, 2, 4, 8],
                 mlp_ratio=2,
                 s_scale=0.875,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 bias=False,
                 num_refinement_blocks=4
                 ):
        super(SSFNR, self).__init__()
        out_channels = in_chans
        self.patch_embed = OverlapPatchEmbed(in_chans, dim)
        self.encoder_level1 = nn.ModuleList([SFB(dim=dim,
                                                        num_heads=num_heads[0],
                                                        reso=img_size,
                                                        mlp_ratio=mlp_ratio,
                                                        s_scale=s_scale,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        split_size=split_size_0[0],
                                                        drop=drop_rate,
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=drop_path_rate,
                                                        act_layer=act_layer,
                                                        norm_layer=norm_layer,
                                                        shift_size=0 if (i % 2 == 0) else split_size_0[0] // 2,
                                                        )
                                             for i in range(depth[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.ModuleList([SFB(dim=int(dim * 2 ** 1),
                                                        num_heads=num_heads[1],
                                                        reso=img_size,
                                                        mlp_ratio=mlp_ratio,
                                                        s_scale=s_scale,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        split_size=split_size_0[1],
                                                        drop=drop_rate,
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=drop_path_rate,
                                                        act_layer=act_layer,
                                                        norm_layer=norm_layer,
                                                        shift_size=0 if (i % 2 == 0) else split_size_0[1] // 2,
                                                        )
                                             for i in range(depth[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.ModuleList([SFB(dim=int(dim * 2 ** 2),
                                                        num_heads=num_heads[2],
                                                        reso=img_size,
                                                        mlp_ratio=mlp_ratio,
                                                        s_scale=s_scale,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        split_size=split_size_0[2],
                                                        drop=drop_rate,
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=drop_path_rate,
                                                        act_layer=act_layer,
                                                        norm_layer=norm_layer,
                                                        shift_size=0 if (i % 2 == 0) else split_size_0[2] // 2,
                                                        )
                                             for i in range(depth[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.ModuleList([SFB(dim=int(dim * 2 ** 3),
                                                num_heads=num_heads[3],
                                                reso=img_size,
                                                mlp_ratio=mlp_ratio,
                                                s_scale=s_scale,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                split_size=split_size_0[3],
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=drop_path_rate,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer,
                                                shift_size=0 if (i % 2 == 0) else split_size_0[3] // 2,
                                                )
                                     for i in range(depth[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([SFB(dim=int(dim * 2 ** 2),
                                                        num_heads=num_heads[2],
                                                        reso=img_size,
                                                        mlp_ratio=mlp_ratio,
                                                        s_scale=s_scale,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        split_size=split_size_0[2],
                                                        drop=drop_rate,
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=drop_path_rate,
                                                        act_layer=act_layer,
                                                        norm_layer=norm_layer,
                                                        shift_size=0 if (i % 2 == 0) else split_size_0[2] // 2,
                                                        )
                                             for i in range(depth[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([SFB(dim=int(dim * 2 ** 1),
                                                        num_heads=num_heads[1],
                                                        reso=img_size,
                                                        mlp_ratio=mlp_ratio,
                                                        s_scale=s_scale,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        split_size=split_size_0[1],
                                                        drop=drop_rate,
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=drop_path_rate,
                                                        act_layer=act_layer,
                                                        norm_layer=norm_layer,
                                                        shift_size=0 if (i % 2 == 0) else split_size_0[1] // 2,
                                                        )
                                             for i in range(depth[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.ModuleList([SFB(dim=int(dim * 2 ** 1),
                                                        num_heads=num_heads[0],
                                                        reso=img_size,
                                                        mlp_ratio=mlp_ratio,
                                                        s_scale=s_scale,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        split_size=split_size_0[0],
                                                        drop=drop_rate,
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=drop_path_rate,
                                                        act_layer=act_layer,
                                                        norm_layer=norm_layer,
                                                        shift_size=0 if (i % 2 == 0) else split_size_0[0] // 2,
                                                        )
                                             for i in range(depth[0])])

        self.refinement = nn.ModuleList([SFB(dim=int(dim * 2 ** 1),
                                                    num_heads=num_heads[0],
                                                    reso=img_size,
                                                    mlp_ratio=mlp_ratio,
                                                    s_scale=s_scale,
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    split_size=split_size_0[0],
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=drop_path_rate,
                                                    act_layer=act_layer,
                                                    norm_layer=norm_layer,
                                                    shift_size=0 if (i % 2 == 0) else split_size_0[0] // 2,
                                                    )
                                         for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4,
                                   w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c")
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c")
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1