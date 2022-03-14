import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.optim import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *

from lib.backbones.ResNet import resnet50
from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinT, SwinS, SwinB, SwinL

class InSPyRationV3(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(InSPyRationV3, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=base_size, stage=4)

        self.decoder = PAA_d(self.depth, base_size=base_size, stage=2)

        self.attention0 = ASCA(self.depth    , depth, base_size=base_size, stage=0, lmap_in=True)
        self.attention1 = ASCA(self.depth * 2, depth, base_size=base_size, stage=1, lmap_in=True)
        self.attention2 = ASCA(self.depth * 2, depth, base_size=base_size, stage=2              )

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        self.reduce = conv(4, 3, 7)
        
        self.forward = self.forward_inference
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(InSPyRationV3, self).cuda()
        return self
    
    def train(self):
        self = super(InSPyRationV3, self).train(mode=True)
        self.forward = self.forward_train
        return self
    
    def eval(self):
        self = super(InSPyRationV3, self).train(mode=False)
        self.forward = self.forward_inference
        return self
    
    def forward_encoder(self, x):
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32
        
        return x1, x2, x3, x4, x5
    
    def forward_decoder(self, xs, f=None, d=None):
        x1, x2, x3, x4, x5 = xs
        B, _, H, W = x1.shape

        f3, d3 = self.decoder(x3, x4, x5) #16

        if f is not None:
            f3 = f
        f3 = self.res(f3, (H,  W))
        
        if d is not None:
            d3 = self.res(d, (H // 2, W // 2))
            
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        d2 = self.pyr.rec(d3.detach(), p2) #4

        x1 = self.res(x1, (H * 2, W * 2))
        f2 = self.res(f2, (H * 2, W * 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach()) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H * 4, W * 4))
        f0, p0 = self.attention0(f1, d1.detach(), p1.detach()) #2
        d0 = self.pyr.rec(d1.detach(), p0) #2
        
        return d3, d2, d1, d0, p2, p1, p0, f3, f2, f1, f0
        
    
    def forward_train(self, sample):
        x = sample['image']
        m = sample['depth']
        
        B, _, H, W = x.shape
    
        x = self.reduce(torch.cat([x, m], dim=1))
        x1, x2, x3, x4, x5 = self.forward_encoder(x)
        d3, d2, d1, d0, p2, p1, p0, _, _, _, _ = self.forward_decoder([x1, x2, x3, x4, x5])
            
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

        sample['pred'] = d0
        sample['loss'] = loss
        sample['gaussian'] = [d3, d2, d1, d0]
        sample['laplacian'] = [p2, p1, p0]
        return sample
    
    def forward_inference(self, sample):
        x = sample['image']
        m = sample['depth']
        B, _, H, W = x.shape
    
        x = self.reduce(torch.cat([x, m], dim=1))
        xs = self.forward_encoder(self.res(x, self.base_size))
        dp = self.forward_decoder(xs)
        
        xs = self.forward_encoder(x)
        dp = self.forward_decoder(xs, f=None, d=dp[3])
        
        d3, d2, d1, d0, p2, p1, p0, _, _, _, _ = dp
        
        loss = 0

        sample['pred'] = d0
        sample['loss'] = loss
        sample['gaussian'] = [d3, d2, d1, d0]
        sample['laplacian'] = [p2, p1, p0]
        return sample
    
    # def forward(self, sample):
    #     x = sample['image']
    #     B, _, H, W = x.shape
    
    #     level_count = int(min(H / self.base_size[0], W / self.base_size[1]))
    #     print(level_count)
            
    #     dp = None
    #     for scale in reversed(range(level_count)):
    #         xs = self.forward_encoder(self.res(x, (H // (scale + 1), W // (scale + 1))))
    #         dp = self.forward_decoder(xs, dp[3] if dp is not None else None)
        
    #     d3, d2, d1, d0, p2, p1, p0 = dp
        
    #     loss = 0

    #     sample['pred'] = d0
    #     sample['loss'] = loss
    #     sample['gaussian'] = [d3, d2, d1, d0]
    #     sample['laplacian'] = [p2, p1, p0]
    #     return sample
    
    

# class InSPyRationV3D(InSPyRationV3):
#     def __init__(self, backbone, in_channels, depth=64, base_size=384, **kwargs):
#         super(InSPyRationV3D, self).__init__(backbone, in_channels, depth, base_size, **kwargs)
#         self.reduce = conv(4, 3, 3)
        
#     def forward(self, sample):
#         x = torch.cat([sample['image'], sample['depth']], dim=1)
#         x = self.reduce(x)
        
#         sample['image'] = x
#         return super(InSPyRationV3D, self).forward(sample)
    
    
def InSPyRationV3_ResNet50(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(resnet50(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size)
    
def InSPyRationV3_Res2Net50(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyRationV3_Res2Net101(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyRationV3_SwinS(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def InSPyRationV3_SwinT(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def InSPyRationV3_SwinB(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def InSPyRationV3_SwinL(depth, pretrained, base_size, **kwargs):
    return InSPyRationV3(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)


# def InSPyRationV3D_Res2Net50(depth, pretrained, base_size, **kwargs):
#     return InSPyRationV3D(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def InSPyRationV3D_Res2Net101(depth, pretrained, base_size, **kwargs):
#     return InSPyRationV3D(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def InSPyRationV3D_SwinS(depth, pretrained, base_size, **kwargs):
#     return InSPyRationV3D(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

# def InSPyRationV3D_SwinT(depth, pretrained, base_size, **kwargs):
#     return InSPyRationV3D(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
# def InSPyRationV3D_SwinB(depth, pretrained, base_size, **kwargs):
#     return InSPyRationV3D(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

# def InSPyRationV3D_SwinL(depth, pretrained, base_size, **kwargs):
#     return InSPyRationV3D(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)