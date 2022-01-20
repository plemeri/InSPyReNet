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

class InSPyReRocket(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=384, **kwargs):
        super(InSPyReRocket, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        
        self.reduce = conv(4, 3, 3)
        
        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=base_size, stage=4)

        self.decoder = PAA_d(self.depth, base_size=base_size, stage=2)

        self.attention0 = DACA(self.depth    , depth, lmap_in=True)
        self.attention1 = DACA(self.depth * 2, depth, lmap_in=True)
        self.attention2 = DACA(self.depth * 2, depth)

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(InSPyReRocket, self).cuda()
        return self
    
    def forward(self, sample):
        x = sample['image']
        dh = sample['depth']
        B, _, H, W = x.shape    
        
        dh1 = self.pyr.down(dh)
        dh2 = self.pyr.down(dh1)
        dh3 = self.pyr.down(dh2)
        
        x = torch.cat([x, dh], dim=1)
        x = self.reduce(x)
                    
        B, _, H, W = x.shape
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder(x3, x4, x5) #16

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach(), dh3)
        d2 = self.pyr.rec(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach(), dh2, p2.detach()) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1.detach(), dh1, p1.detach()) #2
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
            closs += self.loss_fn(self.des(d0, (H, W)), self.des(y,  (H, W)))
            
            loss = ploss + closs

        else:
            loss = 0

        sample['pred'] = d0
        sample['loss'] = loss
        sample['gaussian'] = [d3, d2, d1, d0]
        sample['laplacian'] = [p2, p1, p0]
        return sample
    
def InSPyReRocket_Res2Net50(depth, pretrained, base_size, **kwargs):
    return InSPyReRocket(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReRocket_Res2Net101(depth, pretrained, base_size, **kwargs):
    return InSPyReRocket(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReRocket_SwinS(depth, pretrained, base_size, **kwargs):
    return InSPyReRocket(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def InSPyReRocket_SwinT(depth, pretrained, base_size, **kwargs):
    return InSPyReRocket(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def InSPyReRocket_SwinB(depth, pretrained, base_size, **kwargs):
    return InSPyReRocket(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def InSPyReRocket_SwinL(depth, pretrained, base_size, **kwargs):
    return InSPyReRocket(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)