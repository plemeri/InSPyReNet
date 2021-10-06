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
    # res2net based encoder decoder
    def __init__(self, channels=64, pretrained=True):
        super(InSPyReNet_SwinB, self).__init__()
        # self.backbone = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)
        self.backbone = SwinB(pretrained=pretrained)

        self.context1 = PAA_e(128, channels)
        self.context2 = PAA_e(256, channels)
        self.context3 = PAA_e(512, channels)
        self.context4 = PAA_e(1024, channels)
        self.context5 = PAA_e(1024, channels)

        self.decoder = PAA_d(channels)

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

    def forward(self, sample):
        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None

        B, _, H, W = x.shape # (b, 32H, 32W, 3)

        x1 = self.backbone.stem(x) # 8h 8w
        x2 = self.backbone.layers[0](x1) # 4h 4w
        x3 = self.backbone.layers[1](x2) # 2h 2w
        x4 = self.backbone.layers[2](x3) # h w
        x5 = self.backbone.layers[3](x4) # hw

        x1 = x1.view(B, H // 4, W // 4, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.view(B, H // 8, W // 8, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x3.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2).contiguous()
        x4 = x4.view(B, H // 32, W // 32, -1).permute(0, 3, 1, 2).contiguous()
        x5 = x5.view(B, H // 32, W // 32, -1).permute(0, 3, 1, 2).contiguous()

        x1 = self.context1(x1)
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)
        x5 = self.context5(x5)

        f3, d3 = self.decoder(x5, x4, x3) # 2h 2w

        f2, p2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), d3.detach()) 
        d2 = self.inspyre.rec(d3.detach(), p2) # 4h 4w

        f1, p1 = self.attention1(torch.cat([x1, self.ret(f2, x1)], dim=1), d2.detach(), p2.detach())
        d1 = self.inspyre.rec(d2.detach(), p1) # 8h 8w

        _, p = self.attention(self.res(f1, (H // 2, W // 2)), d1.detach(), p1.detach())
        d = self.inspyre.rec(d1.detach(), p) # 32H X 32W

        if y is not None:       
            py1, y1 = self.spyd.dec(y)
            py2, y2 = self.spyd.dec(y1)
            py3, y3 = self.spyd.dec(y2)
            py4, y4 = self.spyd.dec(y3)
            py5, y5 = self.spyd.dec(y4)

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

            d =  self.res(d, (H, W))

            y4 = self.des(y4, (H, W))
            y3 = self.des(y3, (H, W))
            y2 = self.des(y2, (H, W))

            closs =  self.loss_fn(d3, y4)
            closs += self.loss_fn(d2, y3)
            closs += self.loss_fn(d1, y2)
            closs += self.loss_fn(d,  y)
            
            loss = ploss1 + ploss2 + ploss3 + closs

            debug = [p, d1, p1, d2, p2, d3, y]
        else:
            d =  self.res(d, (H, W))
            loss = 0
            debug = []

        return {'pred': d, 'loss': loss, 'debug': debug}