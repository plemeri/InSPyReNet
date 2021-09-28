import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.SwinTransformerSS import SwinB

class InSPyReNet_SwinBSS(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=64, pretrained=True):
        super(InSPyReNet_SwinBSS, self).__init__()
        # self.backbone = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)
        self.backbone = SwinB(pretrained=pretrained)

        self.context1 = PAA_e(128, channels)
        self.context2 = PAA_e(128, channels)
        self.context3 = PAA_e(256, channels)
        self.context4 = PAA_e(512, channels)
        self.context5 = PAA_e(1024, channels)

        self.decoder = PAA_ds(channels)

        self.attention =  ASCA(channels    , channels, lmap_in=True)
        self.attention1 = ASCA(channels * 2, channels, lmap_in=True)
        self.attention2 = ASCA(channels * 2, channels)

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.inspyre = InSPyRe(7, 1)
        self.spyd = SPyD(7, 1)

    def forward(self, x, y=None):
        B, _, H, W = x.shape # (b, 32H, 32W, 3)
        x1, x2, x3, x4, x5 = self.backbone(x)

        x1 = self.context1(x1.contiguous())
        x2 = self.context2(x2.contiguous())
        x3 = self.context3(x3)
        x4 = self.context4(x4)
        x5 = self.context5(x5)

        f4, d4 = self.decoder(x5, x4, x3) # 2h 2w
        # print(x3.shape, x2.shape, x1.shape)

        f3, p3 = self.attention2(torch.cat([x3, self.ret(f4, x3)], dim=1), d4.detach())
        d3 = self.inspyre.rec(d4.detach(), p3) # os 8

        f2, p2 = self.attention1(torch.cat([x2, self.ret(f3, x2)], dim=1), d3.detach(), p3.detach())
        d2 = self.inspyre.rec(d3.detach(), p2) # os 4

        _, p1 = self.attention(self.res(f2, (H // 2, W // 2)), d2.detach(), p2.detach())
        d1 = self.inspyre.rec(d2.detach(), p1) # os 2

        if y is not None:       
            py1, y1 = self.spyd.dec(y)
            py2, y2 = self.spyd.dec(y1)
            py3, y3 = self.spyd.dec(y2)
            py4, y4 = self.spyd.dec(y3)
            py5, y5 = self.spyd.dec(y4)

            dd4 = self.inspyre.down(d3)
            dd3 = self.inspyre.down(d2)
            dd2 = self.inspyre.down(d1)

            d4 = self.des(d4, (H, W))
            d3 = self.des(d3, (H, W))
            d2 = self.des(d2, (H, W))

            dd4 = self.des(dd4, (H, W))
            dd3 = self.des(dd3, (H, W))
            dd2 = self.des(dd2, (H, W))

            ploss1 = self.pyramidal_consistency_loss_fn(d4, dd4.detach()) * 0.0001
            ploss2 = self.pyramidal_consistency_loss_fn(d3, dd3.detach()) * 0.0001
            ploss3 = self.pyramidal_consistency_loss_fn(d2, dd2.detach()) * 0.0001

            d1 =  self.res(d1, (H, W))

            y4 = self.des(y4, (H, W))
            y3 = self.des(y3, (H, W))
            y2 = self.des(y2, (H, W))

            closs =  self.loss_fn(d4, y4)
            closs += self.loss_fn(d3, y3)
            closs += self.loss_fn(d2, y2)
            closs += self.loss_fn(d1,  y)
            
            loss = ploss1 + ploss2 + ploss3 + closs

            debug = [p1, p2, d2, p3, d3, d4, y]
        else:
            d1 =  self.res(d1, (H, W))
            loss = 0
            debug = []

        return {'pred': d1, 'loss': loss, 'debug': debug}