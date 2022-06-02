import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from operator import xor
from typing import Optional

from lib.modules.layers import *
from utils.misc import *

from lib.modules.layers import FFC_BN_ACT

class reverse_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(reverse_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for _ in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * (torch.sigmoid(map)) + 1
        
        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class simple_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(simple_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for _ in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        amap = torch.sigmoid(map)
        
        x = amap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out
    
class UACA(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = .5 - torch.abs(p) # confusion area

        prob = torch.cat([fg, bg, cg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        
        return x, out
class SICA(nn.Module):
    def __init__(self, in_channel, out_channel=1, depth=64, base_size=None, stage=None, lmap_in=False):
        super(SICA, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.lmap_in = lmap_in
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        
        self.conv_query = nn.Sequential(conv(in_channel, depth, 3, relu=True),
                                        conv(depth, depth, 3, relu=True))
        self.conv_key   = nn.Sequential(conv(in_channel, depth, 1, relu=True),
                                        conv(depth, depth, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, depth, 1, relu=True),
                                        conv(depth, depth, 1, relu=True))

        if self.lmap_in is True:
            self.ctx = 5
        else:
            self.ctx = 3

        self.conv_out1 = conv(depth, depth, 3, relu=True)
        self.conv_out2 = conv(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = conv(depth, depth, 3, relu=True)
        self.conv_out4 = conv(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))
        
        if self.lmap_in is True:
            self.lthreshold = Parameter(torch.tensor([0.5]))

    def forward(self, x, smap, lmap: Optional[torch.Tensor]=None):
        assert not xor(self.lmap_in is True, lmap is not None)
        b, c, h, w = x.shape
        
        # compute class probability
        smap = F.interpolate(smap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = self.threshold - torch.abs(p) # confusion area

        if self.lmap_in is True and lmap is not None:
            lmap = F.interpolate(lmap, size=x.shape[-2:], mode='bilinear', align_corners=False)
            lmap = torch.sigmoid(lmap)
            lp = lmap - self.lthreshold
            fp = torch.clip(lp, 0, 1) # foreground
            bp = torch.clip(-lp, 0, 1) # background

            prob = [fg, bg, cg, fp, bp]
        else:
            prob = [fg, bg, cg]

        prob = torch.cat(prob, dim=1)

        # reshape feature & prob
        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w        
        
        f = F.interpolate(x, size=shape, mode='bilinear', align_corners=False).view(b, shape_mul, -1)
        prob = F.interpolate(prob, size=shape, mode='bilinear', align_corners=False).view(b, self.ctx, shape_mul)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.depth ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)
        
        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out
    
class FFC_SICA(nn.Module):
    def __init__(self, in_channel, out_channel=1, depth=64, base_size=None, stage=None, lmap_in=False):
        super(FFC_SICA, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.lmap_in = lmap_in
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        self.ratio_gin = (5 - stage) / 5
        self.ratio_gout = stage / 5
        
        self.conv_query = nn.Sequential(FFC_BN_ACT(in_channel, depth, 3, padding=1, activation_layer=nn.ReLU),
                                        FFC_BN_ACT(depth, depth, 3, relu=True))
        self.conv_key   = nn.Sequential(conv(in_channel, depth, 1, relu=True),
                                        conv(depth, depth, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, depth, 1, relu=True),
                                        conv(depth, depth, 1, relu=True))

        if self.lmap_in is True:
            self.ctx = 5
        else:
            self.ctx = 3

        self.conv_out1 = conv(depth, depth, 3, relu=True)
        self.conv_out2 = conv(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = conv(depth, depth, 3, relu=True)
        self.conv_out4 = conv(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))
        
        if self.lmap_in is True:
            self.lthreshold = Parameter(torch.tensor([0.5]))

    def forward(self, x, smap, lmap: Optional[torch.Tensor]=None):
        assert not xor(self.lmap_in is True, lmap is not None)
        b, c, h, w = x.shape
        
        # compute class probability
        smap = F.interpolate(smap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = self.threshold - torch.abs(p) # confusion area

        if self.lmap_in is True and lmap is not None:
            lmap = F.interpolate(lmap, size=x.shape[-2:], mode='bilinear', align_corners=False)
            lmap = torch.sigmoid(lmap)
            lp = lmap - self.lthreshold
            fp = torch.clip(lp, 0, 1) # foreground
            bp = torch.clip(-lp, 0, 1) # background

            prob = [fg, bg, cg, fp, bp]
        else:
            prob = [fg, bg, cg]

        prob = torch.cat(prob, dim=1)

        # reshape feature & prob
        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w        
        
        f = F.interpolate(x, size=shape, mode='bilinear', align_corners=False).view(b, shape_mul, -1)
        prob = F.interpolate(prob, size=shape, mode='bilinear', align_corners=False).view(b, self.ctx, shape_mul)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.depth ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)
        
        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out

class SIOC(nn.Module):
    def __init__(self, in_channel, out_channel, depth, base_size=None, stage=None):
        super(SIOC, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        
        self.conv_query = nn.Sequential(conv(in_channel, depth, 3, relu=True),
                                        conv(depth, depth, 3, relu=True))
        self.conv_key   = nn.Sequential(conv(in_channel, depth, 1, relu=True),
                                        conv(depth, depth, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, depth, 1, relu=True),
                                        conv(depth, depth, 1, relu=True))

        self.conv_out1 = conv(depth, depth, 3, relu=True)
        self.conv_out2 = conv(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = conv(depth, depth, 3, relu=True)
        self.conv_out4 = conv(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))
        
    def forward(self, x, smap):
        b, c, h, w = x.shape
        
        # compute class probability
        smap = F.interpolate(smap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background

        prob = [fg, bg]

        prob = torch.cat(prob, dim=1)

        # reshape feature & prob
        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w        
        
        f = F.interpolate(x, size=shape, mode='bilinear', align_corners=False).view(b, shape_mul, -1)
        prob = F.interpolate(prob, size=shape, mode='bilinear', align_corners=False).view(b, 2, shape_mul)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.depth ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)
        
        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return out
