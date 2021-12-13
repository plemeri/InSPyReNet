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

class InSPyReNetV5(nn.Module):
    def __init__(self, backbone, in_channels, depth=64):
        super(InSPyReNetV5, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        
        self.reduce = conv(4, 3, 3)
        
        self.context1 = PAA_e(self.in_channels[0], self.depth)
        self.context2 = PAA_e(self.in_channels[1], self.depth)
        self.context3 = PAA_e(self.in_channels[2], self.depth)
        self.context4 = PAA_e(self.in_channels[3], self.depth)
        self.context5 = PAA_e(self.in_channels[4], self.depth)

        self.decoder = PAA_d(self.depth)

        self.attention0_1 = Attn(self.depth    , depth, decoder=False)
        self.attention1_1 = Attn(self.depth * 2, depth, decoder=False)
        self.attention2_1 = Attn(self.depth * 2, depth, decoder=False)
        
        self.attention0_2 = Attn(self.depth    , depth, decoder=True)
        self.attention1_2 = Attn(self.depth * 2, depth, decoder=True)
        self.attention2_2 = Attn(self.depth * 2, depth, decoder=True)

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(InSPyReNetV5, self).cuda()
        return self
    
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
            dh = sample['depth']
        else:
            x, dh = sample
            
        x = torch.cat([x, dh], dim=1)
        x = self.reduce(x)
        
        dh1 = self.pyr.down(dh)
        dh2 = self.pyr.down(dh1)
        dh3 = self.pyr.down(dh2)
            
        B, _, H, W = x.shape
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder(x5, x4, x3) #16

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, _ = self.attention2_1(torch.cat([x2, f3], dim=1), d3.detach())
        f2, p2 = self.attention2_2(torch.cat([f2, f3], dim=1), dh3)
        d2 = self.pyr.rec(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, _ = self.attention1_1(torch.cat([x1, f2], dim=1), d2.detach()) #2
        f1, p1 = self.attention1_2(torch.cat([f1, f2], dim=1), dh2) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H, W))
        f1, _ = self.attention0_1(f1, d1.detach()) #2
        f1, p0 = self.attention0_2(f1, dh1) #2
        d0 = self.pyr.rec(d1.detach(), p0) #2
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            y1 = self.pyr.down(y)
            y2 = self.pyr.down(y1)
            y3 = self.pyr.down(y2)

            ploss =  self.pyramidal_consistency_loss_fn(self.des(d3, (H, W)), self.des(self.pyr.down(d2), (H, W)).detach()) * 0.0001
            ploss += self.pyramidal_consistency_loss_fn(self.des(d2, (H, W)), self.des(self.pyr.down(d1), (H, W)).detach()) * 0.0001
            ploss += self.pyramidal_consistency_loss_fn(self.des(d1, (H, W)), self.des(self.pyr.down(d0), (H, W)).detach()) * 0.0001
            
            closs =  self.loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
            closs += self.loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
            closs += self.loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
            closs += self.loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
            
            loss = ploss + closs

        else:
            loss = 0

        if type(sample) == dict:
            return {'pred': d0, 
                    'loss': loss, 
                    'gaussian': [d3, d2, d1, d0], 
                    'laplacian': [p2, p1, p0]}
        
        else:
            return d0
    
    
def InSPyReNetV5_Res2Net50(depth, pretrained):
    return InSPyReNetV5(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetV5_Res2Net101(depth, pretrained):
    return InSPyReNetV5(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetV5_SwinS(depth, pretrained):
    return InSPyReNetV5(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth)

def InSPyReNetV5_SwinT(depth, pretrained):
    return InSPyReNetV5(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth)
    
def InSPyReNetV5_SwinB(depth, pretrained):
    return InSPyReNetV5(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth)

def InSPyReNetV5_SwinL(depth, pretrained):
    return InSPyReNetV5(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth)