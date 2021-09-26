import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.SwinTransformer import SwinB_224

class EInSPyReNet_SwinB(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=64, pretrained=True):
        super(EInSPyReNet_SwinB, self).__init__()
        # self.backbone = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)
        self.backbone = SwinB_224(pretrained=pretrained)

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

    @staticmethod
    def patch(x, patch_size=256):
        b, c, h, w = x.shape
        unfold  = nn.Unfold(kernel_size=(patch_size,) * 2, stride=patch_size // 2)

        patches = unfold(x)
        patches = patches.reshape(c, patch_size, patch_size, -1).contiguous().permute(3, 0, 1, 2)
        
        return patches, (b, c, h, w)

    @staticmethod
    def stitch(patches, target_shape, patch_size=256):
        b, c, h, w = target_shape
        fold = nn.Fold(output_size=(h, w), kernel_size=(patch_size,) * 2, stride=patch_size // 2)
        unfold  = nn.Unfold(kernel_size=(patch_size,) * 2, stride=patch_size // 2)

        patches = patches.permute(1, 2, 3, 0).reshape(b, c * patch_size ** 2, patches.shape[0] // b)

        weight = torch.ones(*target_shape).to(patches.device)
        weight  = unfold(weight)
        
        out = fold(patches) / fold(weight)

        return out
        

    def forward(self, x, y=None):
        B, _, H, W = x.shape # (b, 32H, 32W, 3)
        px, _ = self.patch(x, 224)
        pB, _, pH, pW = px.shape

        px1 = self.backbone.stem(px) # 8h 8w
        px2 = self.backbone.layers[0](px1) # 4h 4w
        px3 = self.backbone.layers[1](px2) # 2h 2w
        px4 = self.backbone.layers[2](px3) # h w
        px5 = self.backbone.layers[3](px4) # hw

        px1 = px1.view(pB, pH // 4,  pW // 4, -1).permute(0, 3, 1, 2).contiguous()
        px2 = px2.view(pB, pH // 8,  pW // 8, -1).permute(0, 3, 1, 2).contiguous()
        px3 = px3.view(pB, pH // 16, pW // 16, -1).permute(0, 3, 1, 2).contiguous()
        px4 = px4.view(pB, pH // 32, pW // 32, -1).permute(0, 3, 1, 2).contiguous()
        px5 = px5.view(pB, pH // 32, pW // 32, -1).permute(0, 3, 1, 2).contiguous()

        px1 = self.context1(px1)
        px2 = self.context2(px2)
        px3 = self.context3(px3)
        px4 = self.context4(px4)
        px5 = self.context5(px5)

        f3, pd3 = self.decoder(px5, px4, px3) # 2h 2w

        f2, pl2 = self.attention2(torch.cat([px2, self.ret(f3, px2)], dim=1), pd3.detach()) 
        pd2 = self.inspyre.rec(pd3.detach(), pl2) # 4h 4w

        f1, pl1 = self.attention1(torch.cat([px1, self.ret(f2, px1)], dim=1), pd2.detach(), pl2.detach())
        pd1 = self.inspyre.rec(pd2.detach(), pl1) # 8h 8w

        _, pl = self.attention(self.res(f1, (pH // 2, pW // 2)), pd1.detach(), pl1.detach())
        # pd = self.inspyre.rec(pd1.detach(), pl) # 32H X 32W

        d3 = self.stitch(pd3, (B, 1, H // 16, W // 16), patch_size=224 // 16)
        l2 = self.stitch(pl2, (B, 1, H // 8, W // 8), patch_size=224 // 8)

        d2 = self.inspyre.rec(d3, l2)
        l1 = self.stitch(pl1, (B, 1, H // 4, W // 4), patch_size=224 // 4)

        d1 = self.inspyre.rec(d2, l1)
        l = self.stitch(pl, (B, 1, H // 2, W // 2), patch_size=224 // 2)
        
        d = self.inspyre.rec(d1, l)

        if y is not None:
            ly1, y1 = self.spyd.dec(y)
            ly2, y2 = self.spyd.dec(y1)
            ly3, y3 = self.spyd.dec(y2)
            ly4, y4 = self.spyd.dec(y3)
            ly5, y5 = self.spyd.dec(y4)

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

            debug = [l, d1, l1, d2, l2, d3, y]
        else:
            d =  self.res(d, (H, W))
            loss = 0
            debug = []

        return {'pred': d, 'loss': loss, 'debug': debug}