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
from lib.backbones.FFCResNet import ffc_resnet50

class InSPyReNet(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(InSPyReNet, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=self.base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=self.base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=self.base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=self.base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=self.base_size, stage=4)

        self.decoder = PAA_d(self.depth * 3, 1, depth=self.depth, base_size=base_size, stage=2)

        self.attention0 = SICA(self.depth    , 1, depth=self.depth, base_size=self.base_size, stage=0, lmap_in=True)
        self.attention1 = SICA(self.depth * 2, 1, depth=self.depth, base_size=self.base_size, stage=1, lmap_in=True)
        self.attention2 = SICA(self.depth * 2, 1, depth=self.depth, base_size=self.base_size, stage=2              )
        
        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + \
            focal_tversky_loss_with_logits(x, y, alpha=0.2, beta=0.8, gamma=2, reduction='mean')
        self.pc_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(InSPyReNet, self).cuda()
        return self
    
    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder([x3, x4, x5]) #16

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        d2 = self.pyr.rec(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach()) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1.detach(), p1.detach()) #2
        d0 = self.pyr.rec(d1.detach(), p0) #2
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            y1 = self.pyr.down(y)
            y2 = self.pyr.down(y1)
            y3 = self.pyr.down(y2)

            ploss =  self.pc_loss_fn(self.des(d3, (H, W)), self.des(self.pyr.down(d2), (H, W)).detach()) * 0.0001
            ploss += self.pc_loss_fn(self.des(d2, (H, W)), self.des(self.pyr.down(d1), (H, W)).detach()) * 0.0001
            ploss += self.pc_loss_fn(self.des(d1, (H, W)), self.des(self.pyr.down(d0), (H, W)).detach()) * 0.0001
            
            closs =  self.sod_loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
            closs += self.sod_loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
            closs += self.sod_loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
            closs += self.sod_loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
            
            loss = ploss + closs

        else:
            loss = 0
            
        pred = torch.sigmoid(d0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        sample['gaussian'] = [d3, d2, d1, d0]
        sample['laplacian'] = [p2, p1, p0]
        return sample
    
def InSPyReNet_FFCResNet50(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(ffc_resnet50(pretrained=pretrained, ratio=0.25), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)
    
def InSPyReNet_Res2Net50(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReNet_Res2Net101(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReNet_SwinS(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def InSPyReNet_SwinT(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def InSPyReNet_SwinB(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def InSPyReNet_SwinL(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)

# class InSPyReNetD(InSPyReNet):
#     def __init__(self, backbone, in_channels, depth=64, base_size=384, **kwargs):
#         super(InSPyReNetD, self).__init__(backbone, in_channels, depth, base_size, **kwargs)
#         self.reduce = conv(4, 3, 3)
        
#     def forward(self, sample):
#         x = torch.cat([sample['image'], sample['depth']], dim=1)
#         x = self.reduce(x)
        
#         sample['image'] = x
#         return super(InSPyReNetD, self).forward(sample)
    

# def InSPyReNetD_Res2Net50(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetD(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def InSPyReNetD_Res2Net101(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetD(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def InSPyReNetD_SwinS(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetD(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

# def InSPyReNetD_SwinT(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetD(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
# def InSPyReNetD_SwinB(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetD(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

# def InSPyReNetD_SwinL(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetD(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)