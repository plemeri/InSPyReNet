
from audioop import reverse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.optim import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinT, SwinS, SwinB, SwinL

class RAS(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(RAS, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        self.decoder5 = nn.Sequential(
            conv(in_channels[4], 256, 1),
            conv(256, 256, 5),
            conv(256, 256, 5),
            conv(256, 256, 5),
            conv(256, 1, 1)
        )
        
        self.attention1 = reverse_attention(in_channels[0], self.depth, 2)
        self.attention2 = reverse_attention(in_channels[1], self.depth, 2)
        self.attention3 = reverse_attention(in_channels[2], self.depth, 2)
        self.attention4 = reverse_attention(in_channels[3], self.depth, 2)

        self.loss_fn = bce_loss

        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(RAS, self).cuda()
        return self
    
    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        
        d5 = self.decoder5(x5)

        p4 = self.attention4(x4, d5)
        d4 = self.res(d5, (H // 16, W // 16)) + p4

        p3 = self.attention3(x3, d4)
        d3 = self.res(d4, (H // 8, W // 8)) + p3

        x2 = self.res(x2, (H // 2, W // 2))
        p2 = self.attention2(x2, d3)
        d2 = self.res(d3, (H // 4, W // 4)) + p2

        x1 = self.res(x1, (H, W))
        p1 = self.attention1(x1, d2) #2
        d1 = self.res(d2, (H // 2, W // 2)) + p1
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            loss  = self.loss_fn(self.res(d5, (H, W)), y)
            loss += self.loss_fn(self.res(d4, (H, W)), y)
            loss += self.loss_fn(self.res(d3, (H, W)), y)
            loss += self.loss_fn(self.res(d2, (H, W)), y)
            loss += self.loss_fn(self.res(d1, (H, W)), y)
            
        else:
            loss = 0
            
        pred = torch.sigmoid(d1)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        sample['gaussian'] = [d5, d4, d3, d2, d1]
        sample['laplacian'] = [p4, p3, p2, p1]
        return sample
    

    
def RAS_Res2Net50(depth, pretrained, base_size, **kwargs):
    return RAS(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def RAS_Res2Net101(depth, pretrained, base_size, **kwargs):
    return RAS(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def RAS_SwinS(depth, pretrained, base_size, **kwargs):
    return RAS(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def RAS_SwinT(depth, pretrained, base_size, **kwargs):
    return RAS(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def RAS_SwinB(depth, pretrained, base_size, **kwargs):
    return RAS(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def RAS_SwinL(depth, pretrained, base_size, **kwargs):
    return RAS(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)
