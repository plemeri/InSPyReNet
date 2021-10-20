import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinB

class InSPyReNet(nn.Module):
    def __init__(self, backbone, in_channels, depth=64):
        super(InSPyReNet, self).__init__()
        self.backbone = backbone
        
        self.context1 = PAA_e(in_channels[0], depth)
        self.context2 = PAA_e(in_channels[1], depth)
        self.context3 = PAA_e(in_channels[2], depth)
        self.context4 = PAA_e(in_channels[3], depth)
        self.context5 = PAA_e(in_channels[4], depth)

        self.decoder = PAA_d2(depth)

        self.attention =  ASCA(depth    , depth, lmap_in=True)
        self.attention1 = ASCA(depth * 2, depth, lmap_in=True)
        self.attention2 = ASCA(depth * 2, depth, lmap_in=True)
        self.attention3 = ASCA(depth * 2, depth)

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)


    def forward(self, sample):
        B, _, H, W = sample['image'].shape
        x1, x2, x3, x4, x5 = self.backbone(sample['image'])
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f4, d4 = self.decoder(x5, x4) #16

        f3, p3 = self.attention3(torch.cat([x3, self.res(f4, (H // 8,  W // 8 ))], dim=1), d4.detach()) #8
        d3 = self.pyr.rec(d4.detach(), p3) #8

        f2, p2 = self.attention2(torch.cat([x2, self.res(f3, (H // 4,  W // 4 ))], dim=1), d3.detach(), p3.detach()) #4
        d2 = self.pyr.rec(d3.detach(), p2) #4

        _, p1 = self.attention1(torch.cat([self.res(x1, (H // 2, W // 2)), self.res(f2, (H // 2, W // 2))], dim=1), d2.detach(), p2.detach()) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        d =  self.res(d1, (H, W))

        if sample['gt'] is not None:
            y = sample['gt']
            
            y1 = self.pyr.down(y)
            y2 = self.pyr.down(y1)
            y3 = self.pyr.down(y2)

            ploss =  self.pyramidal_consistency_loss_fn(self.des(d3, (H, W)), self.des(self.pyr.down(d2), (H, W)).detach()) * 0.0001
            ploss += self.pyramidal_consistency_loss_fn(self.des(d2, (H, W)), self.des(self.pyr.down(d1), (H, W)).detach()) * 0.0001
            # ploss += self.pyramidal_consistency_loss_fn(self.des(d1, (H, W)), self.des(self.pyr.down(d), (H, W)).detach()) * 0.0001

            closs =  self.loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
            closs += self.loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
            closs += self.loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
            # closs += self.loss_fn(self.des(d, (H, W)), self.des(y, (H, W)))
            
            loss = ploss + closs

            debug = [p1, d2, p2, d3, y]
        else:
            loss = 0
            debug = [d3, p2, p1]

        return {'pred': d, 'loss': loss, 'debug': debug}
    
    
def InSPyReNet_SwinB(depth, pretrained):
    return InSPyReNet(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth)

def InSPyReNet_Res2Net50(depth, pretrained):
    return InSPyReNet(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)