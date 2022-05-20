from optparse import Option
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

from torch.nn.parameter import Parameter
from typing import Optional
class Pyr:
    def __init__(self, ksize=5, sigma=1, channels=1):
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels

        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)
        
    def cuda(self):
        self.kernel = self.kernel.cuda()
        return self

    def up(self, x):
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.conv2d(x, self.kernel * 4, groups=self.channels, padding=self.ksize // 2)
        return x

    def down(self, x):
        x = F.conv2d(x, self.kernel, groups=self.channels, padding=self.ksize // 2)
        return x[:, :, ::2, ::2]

    def dec(self, x):
        down = self.down(x)
        up = self.up(down)

        if x.shape != up.shape:
            up = F.interpolate(up, x.shape[-2:])

        lap = x - up
        return down, lap

    def rec(self, down, lap):
        down = self.up(down)
        if lap.shape != down:
            lap = F.interpolate(lap, down.shape[-2:], mode='bilinear', align_corners=True)
        return down + lap

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same', bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw', stage_size=None):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        self.stage_size = stage_size

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class self_attn2(nn.Module):
    def __init__(self, in_channels, mode='hw', stage_size=None):
        super(self_attn2, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        self.stage_size = stage_size

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        permute = (0, 1, 2, 3)
        if 'h' == self.mode:
            axis = height
            permute = (0, 1, 3, 2)
        elif 'w' == self.mode:
            axis = width
        elif 'hw' == self.mode:
            axis = height * width

        view = (batch_size, -1, axis)
        print(self.mode, axis, permute, view, x.shape)

        projected_query = self.query_conv(x).permute(*permute).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).permute(*permute).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).permute(*permute).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

def patch(x, patch_size=256, stride: Optional[int]=None):
    b, c, h, w = x.shape
    
    if stride is None:
        stride = patch_size // 2
    assert stride != 0
    assert h // stride != 0
    assert w // stride != 0
    
    ph, pw = (h - (patch_size - 1) - 1) // stride + 1, (w - (patch_size - 1) - 1) // stride + 1
    patches = torch.zeros(b * ph * pw, c, patch_size, patch_size).to(x.device)
    
    for i in range(ph):
        for j in range(pw):
            start = pw * i + j
            end = start + 1
            patches[start:end] = x[:, :, i * stride: i * stride + patch_size, j * stride: j * stride + patch_size]
    return patches, (b, c, h, w)

class Patch(nn.Module):
    def __init__(self, patch_size, stride: Optional[int]=None):
        super(Patch, self).__init__()

        self.patch_size = patch_size
        if stride is None:
            self.stride = patch_size // 2
        else:
            self.stride = stride

    def forward(self, x):
        return patch(x, self.patch_size, self.stride)


def unpatch(patches, target_shape, patch_size=256, stride: Optional[int]=None, factor='max'):
    b, c, h, w = target_shape
    
    if stride is None:
        stride = patch_size // 2
    assert stride != 0
    assert h // stride != 0
    assert w // stride != 0
    
    ph, pw = (h - (patch_size - 1) - 1) // stride + 1, (w - (patch_size - 1) - 1) // stride + 1
    out = - torch.ones(ph * pw, b, c, h, w).to(patches.device) * float('inf')
    
    for i in range(ph):
        for j in range(pw):
            start = pw * i + j
            end = start + 1
            out[start:end, :, :, i * stride:i * stride + patch_size, j * stride: j * stride + patch_size] = patches[start:end]
    
    if factor == 'max':
        out, _ = torch.max(out, dim=0)
    else:
        ind = torch.max(torch.abs(out), )
        out = torch.gather(out, 0, ind.unsqueeze(0)).squeeze(0)
    return out



# def unpatch(patches, target_shape, patch_size=256, stride: Optional[int]=None, indice_map: Optional[torch.Tensor]=None, guide=None):
#     b, c, h, w = target_shape
    
#     if stride is None:
#         stride = patch_size // 2
#     assert stride != 0
#     assert h // stride != 0
#     assert w // stride != 0
    
#     ph, pw = (h - (patch_size - 1) - 1) // stride + 1, (w - (patch_size - 1) - 1) // stride + 1
#     out = - torch.ones(ph * pw, b, c, h, w).to(patches.device) * float('inf')
    
#     for i in range(ph):
#         for j in range(pw):
#             start = pw * i + j
#             end = start + 1
#             out[start:end, :, :, i * stride:i * stride + patch_size, j * stride: j * stride + patch_size] = patches[start:end]
#     if guide is not None:
#         out = torch.cat([out, guide.unsqueeze(0)], dim=0)
    
#     if indice_map is None:
#         out, ind = torch.max(out, dim=0)
#     else:
#         ind = indice_map
#         out = torch.gather(out, 0, ind.unsqueeze(0)).squeeze(0)
#     return out, ind

# def unpatch(patches, target_shape, patch_size=256, stride: Optional[int]=None, indice_map: Optional[torch.Tensor]=None, guide=None):
#     b, c, h, w = target_shape
    
#     if stride is None:
#         stride = patch_size // 2
#     assert stride != 0
#     assert h // stride != 0
#     assert w // stride != 0
    
#     ph, pw = (h - (patch_size - 1) - 1) // stride + 1, (w - (patch_size - 1) - 1) // stride + 1
#     out = - torch.ones(ph * pw, b, c, h, w).to(patches.device) * float('inf')
    
#     for i in range(ph):
#         for j in range(pw):
#             start = pw * i + j
#             end = start + 1
#             out[start:end, :, :, i * stride:i * stride + patch_size, j * stride: j * stride + patch_size] = patches[start:end]
    
#     if indice_map is None:
#         out, ind = torch.max(out, dim=0)
#     else:
#         ind = indice_map
#         _, ind = torch.max(torch.abs(out), dim=0)
#         out = torch.gather(out, 0, ind.unsqueeze(0)).squeeze(0)
#     return out, ind

class UnPatch(nn.Module):
    def __init__(self, patch_size, target_shape, stride: Optional[int]=None):
        super(UnPatch, self).__init__()

        self.patch_size = patch_size
        self.target_shape = target_shape
        
        if stride is None:
            self.stride = patch_size // 2
        else:
            self.stride = stride

    def forward(self, x, indice_map=None):
        return unpatch(x, self.target_shape, self.patch_size, self.stride, indice_map)

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        # ffted = torch.fft.rfft(x, signal_ndim=2, normalized=True)
        ffted = torch.view_as_real(torch.fft.fft2(x, norm='ortho'))
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        # output = torch.fft.irfft(ffted, signal_ndim=2, signal_sizes=r_size[2:], normalized=True)
        output = torch.fft.ifft2(torch.view_as_complex(ffted), norm='ortho')
        output = torch.real(output)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fftconv1 = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fftconv1(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.glconv = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.glconv(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g