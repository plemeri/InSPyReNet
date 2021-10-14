import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.SwinTransformer import SwinB

class InSPyReNet_SwinB(nn.Module):
    def __init__(self, channels=64, pretrained=True):
        super(InSPyReNet_SwinB, self).__init__()
        self.backbone = SwinB(pretrained=pretrained)

        self.context1 = PAA_e(128, channels)
        self.context2 = PAA_e(128, channels)
        self.context3 = PAA_e(256, channels)
        self.context4 = PAA_e(512, channels)
        self.context5 = PAA_e(1024, channels)

        self.decoder = PAA_d(channels)
        
        self.attention =  ASCA(channels    , channels, lmap_in=True)
        self.attention1 = ASCA(channels * 2, channels, lmap_in=True)
        self.attention2 = ASCA(channels * 2, channels, lmap_in=True)
        self.attention3 = ASCA(channels * 2, channels, lmap_in=True)
        self.attention4 = ASCA(channels * 2, channels)

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.inspyre = InSPyRe(7, 1)
        self.spyd = SPyD(7, 1)

    def forward(self, sample):
        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None
            
        B, _, H, W = x.shape
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f5, d5 = self.decoder(x3, x4, x5) #32
        
        f4, p4 = self.attention4(torch.cat([x4, self.res(f5, (H // 16, W // 16))], dim=1), d5.detach()) #16
        d4 = self.inspyre.rec(d5.detach(), p4) #16

        f3, p3 = self.attention3(torch.cat([x3, self.res(f4, (H // 8,  W // 8 ))], dim=1), d4.detach(), p4.detach()) #8
        d3 = self.inspyre.rec(d4.detach(), p3) #8

        f2, p2 = self.attention2(torch.cat([x2, self.res(f3, (H // 4,  W // 4 ))], dim=1), d3.detach(), p3.detach()) #4
        d2 = self.inspyre.rec(d3.detach(), p2) #4

        f1, p1 = self.attention1(torch.cat([self.res(x1, (H // 2, W // 2)), self.res(f2, (H // 2, W // 2))], dim=1), d2.detach(), p2.detach()) #2
        d1 = self.inspyre.rec(d2.detach(), p1) #2

        _, p = self.attention(self.res(f1, (H, W)), d1.detach(), p1.detach())
        d = self.inspyre.rec(d1.detach(), p)

        if y is not None:       
            _, y1 = self.spyd.dec(y)
            _, y2 = self.spyd.dec(y1)
            _, y3 = self.spyd.dec(y2)
            _, y4 = self.spyd.dec(y3)

            dd3 = self.inspyre.down(d2)
            dd2 = self.inspyre.down(d1)
            dd1 = self.inspyre.down(d)

            d3 = self.des(d3, (H, W))
            d2 = self.des(d2, (H, W))
            d1 = self.des(d1, (H, W))

            dd3 = self.des(dd3, (H, W))
            dd2 = self.des(dd2, (H, W))
            dd1 = self.des(dd1, (H, W))

            ploss1 = self.pyramidal_consistency_loss_fn(d3, dd3.detach()) * 0.0001
            ploss2 = self.pyramidal_consistency_loss_fn(d2, dd2.detach()) * 0.0001
            ploss3 = self.pyramidal_consistency_loss_fn(d1, dd1.detach()) * 0.0001

            y3 = self.des(y3, (H, W))
            y2 = self.des(y2, (H, W))
            y1 = self.des(y1, (H, W))

            closs =  self.loss_fn(d3, y3)
            closs += self.loss_fn(d2, y2)
            closs += self.loss_fn(d1, y1)
            closs += self.loss_fn(d,  y)
            
            loss = ploss1 + ploss2 + ploss3 + closs

            debug = [p, d1, p1, d2, p2, d3, y]
        else:
            d =  self.res(d, (H, W))
            loss = 0
            debug = [d3, p2, p1, p]

        return {'pred': d, 'loss': loss, 'debug': debug}