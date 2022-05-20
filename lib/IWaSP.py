import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_wavelets as pwt

from lib.optim import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinT, SwinS, SwinB, SwinL

def safe_logit(x):
    return torch.logit(x - torch.sign(x - .5) * torch.ones_like(x) * torch.finfo(torch.float32).eps)

class IWaSP(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(IWaSP, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        if 'wave' in kwargs.keys():
            self.wave = kwargs['wave']
        else:
            self.wave = 'haar'
        
        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=base_size, stage=4)

        self.decoder = PAA_d(self.depth * 3, depth=self.depth, base_size=base_size, stage=2)

        self.lh_attention0 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=0, lmap_in=True)
        self.lh_attention1 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=1, lmap_in=True)
        self.lh_attention2 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=2              )
    
        self.hl_attention0 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=0, lmap_in=True)
        self.hl_attention1 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=1, lmap_in=True)
        self.hl_attention2 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=2              )
        
        self.hh_attention0 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=0, lmap_in=True)
        self.hh_attention1 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=1, lmap_in=True)
        self.hh_attention2 = SICA(self.depth * 2, 1, depth, base_size=base_size, stage=2              )

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss_with_logits(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.fwt = pwt.DWTForward(J=1, wave=self.wave, mode='per')
        self.iwt = pwt.DWTInverse(wave=self.wave, mode='per')

    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder([x3, x4, x5]) #8
        
        lh_f2, lh2 = self.lh_attention2(torch.cat([x3, f3], dim=1), d3.detach())
        hl_f2, hl2 = self.hl_attention2(torch.cat([x3, f3], dim=1), d3.detach())
        hh_f2, hh2 = self.hh_attention2(torch.cat([x3, f3], dim=1), d3.detach())
        
        w2 = torch.stack([lh2, hl2, hh2], dim=2)
        fw2 = torch.stack([lh_f2, hl_f2, hh_f2], dim=2)
        
        d2 = self.iwt((d3 * 2, [w2]))
        f2 = self.iwt((f3 * 2, [fw2]))

        lh_f1, lh1 = self.lh_attention1(torch.cat([x2, f2], dim=1), d2.detach(), lh2.detach())
        hl_f1, hl1 = self.hl_attention1(torch.cat([x2, f2], dim=1), d2.detach(), hl2.detach())
        hh_f1, hh1 = self.hh_attention1(torch.cat([x2, f2], dim=1), d2.detach(), hh2.detach())
        
        w1 = torch.stack([lh1, hl1, hh1], dim=2)
        fw1 = torch.stack([lh_f1, hl_f1, hh_f1], dim=2)
        
        d1 = self.iwt((d2 * 2, [w1]))
        f1 = self.iwt((f2 * 2, [fw1]))
        
        x1 = self.res(x1, (H // 2, W // 2))

        lh_f0, lh0 = self.lh_attention0(torch.cat([x1, f1], dim=1), d1.detach(), lh1.detach())
        hl_f0, hl0 = self.hl_attention0(torch.cat([x1, f1], dim=1), d1.detach(), hl1.detach())
        hh_f0, hh0 = self.hh_attention0(torch.cat([x1, f1], dim=1), d1.detach(), hh1.detach())
        
        w0 = torch.stack([lh0, hl0, hh0], dim=2)
        # fw0 = torch.stack([lh_f0, hl_f0, hh_f0], dim=2)
        
        d0 = self.iwt((d1 * 2, [w0]))
        # f0 = self.iwt((f1, [fw0]))
        
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            y = safe_logit(y)
            y1, _ = self.fwt(y)
            y1 /= 2
            y2, _ = self.fwt(y1)
            y2 /= 2
            y3, _ = self.fwt(y2)
            y3 /= 2

            # ploss =  self.pyramidal_consistency_loss_fn(self.des(d3, (H, W)), self.des(self.fwt(d2)[0], (H, W)).detach()) * 0.0001
            # ploss += self.pyramidal_consistency_loss_fn(self.des(d2, (H, W)), self.des(self.fwt(d1)[0], (H, W)).detach()) * 0.0001
            # ploss += self.pyramidal_consistency_loss_fn(self.des(d1, (H, W)), self.des(self.fwt(d0)[0], (H, W)).detach()) * 0.0001
            
            closs =  self.loss_fn(self.des(d3, (H, W)), self.des(torch.sigmoid(y3), (H, W)))
            closs += self.loss_fn(self.des(d2, (H, W)), self.des(torch.sigmoid(y2), (H, W)))
            closs += self.loss_fn(self.des(d1, (H, W)), self.des(torch.sigmoid(y1), (H, W)))
            closs += self.loss_fn(self.des(d0, (H, W)), self.des(torch.sigmoid(y ), (H, W)))
            
            loss = closs #ploss + closs

        else:
            loss = 0
            
        pred = torch.sigmoid(d0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
        w2 = torch.split(w2, 1, dim=1)
        w1 = torch.split(w1, 1, dim=1)
        w0 = torch.split(w0, 1, dim=1)

        sample['pred'] = pred
        sample['loss'] = loss
        sample['ll'] = [d3, d2, d1, d0]
        sample['lh'] = [lh2, lh1, lh0]
        sample['hl'] = [hl2, hl1, hl0]
        sample['hh'] = [hh2, hh1, hh0]
        
        return sample
    

    
def IWaSP_Res2Net50(depth, pretrained, base_size, **kwargs):
    return IWaSP(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def IWaSP_Res2Net101(depth, pretrained, base_size, **kwargs):
    return IWaSP(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def IWaSP_SwinS(depth, pretrained, base_size, **kwargs):
    return IWaSP(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def IWaSP_SwinT(depth, pretrained, base_size, **kwargs):
    return IWaSP(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def IWaSP_SwinB(depth, pretrained, base_size, **kwargs):
    return IWaSP(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def IWaSP_SwinL(depth, pretrained, base_size, **kwargs):
    return IWaSP(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)