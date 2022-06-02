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

class InSPyReNetv2(nn.Module):
    def __init__(self, backbone, in_channels, depth=64):
        super(InSPyReNetv2, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        
        self.context1 = PAA_e(self.in_channels[0], self.depth)
        self.context2 = PAA_e(self.in_channels[1], self.depth)
        self.context3 = PAA_e(self.in_channels[2], self.depth)
        self.context4 = PAA_e(self.in_channels[3], self.depth)
        self.context5 = PAA_e(self.in_channels[4], self.depth)

        self.decoder = PAA_d(self.depth)

        self.attention0 = Attn(self.depth    , depth)#, lmap_in=True)
        self.attention1 = Attn(self.depth * 2, depth)#, lmap_in=True)
        self.attention2 = Attn(self.depth * 2, depth)
        
        # self.attention0 = simple_attention(self.depth,     self.depth)
        # self.attention1 = simple_attention(self.depth * 2, self.depth)
        # self.attention2 = simple_attention(self.depth * 2, self.depth)

        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.image_pyramidamid_loss_fn = nn.MSELoss() #lambda x, y: weighted_tversky_bce_lossv2(x, y, alpha=0.2, beta=0.8, gamma=2)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.image_pyramid = ImagePyramid(7, 1)
        
    def cuda(self):
        self.image_pyramid = self.image_pyramid.cuda()
        self = super(InSPyReNetv2, self).cuda()
        return self
    
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
        else:
            x = sample
            
        B, _, H, W = x.shape
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder(x5, x4, x3) #16
        d3 = torch.sigmoid(d3)

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        p2 = torch.tanh(p2)
        d2 = self.image_pyramid.reconstruct(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach())#, p2.detach()) #2
        p1 = torch.tanh(p1)
        
        d1 = self.image_pyramid.reconstruct(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1.detach())#, p1.detach()) #2
        p0 = torch.tanh(p0)
        d0 = self.image_pyramid.reconstruct(d1.detach(), p0) #2
        
        print(p2.min(), p1.min(), p0.min())
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            y1, yp0 = self.image_pyramid.deconstruct(y)
            y2, yp1 = self.image_pyramid.deconstruct(y1)
            y3, yp2 = self.image_pyramid.deconstruct(y2)
            
            ploss =  self.image_pyramidamid_loss_fn(self.des(p2, (H, W)), self.des(yp2, (H, W)))
            ploss += self.image_pyramidamid_loss_fn(self.des(p1, (H, W)), self.des(yp1, (H, W)))
            ploss += self.image_pyramidamid_loss_fn(self.des(p0, (H, W)), self.des(yp0, (H, W)))
            
            closs = self.loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
            
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
    
    
def InSPyReNetv2_Res2Net50(depth, pretrained):
    return InSPyReNetv2(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetv2_Res2Net101(depth, pretrained):
    return InSPyReNetv2(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetv2_SwinS(depth, pretrained):
    return InSPyReNetv2(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth)

def InSPyReNetv2_SwinT(depth, pretrained):
    return InSPyReNetv2(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth)
    
def InSPyReNetv2_SwinB(depth, pretrained):
    return InSPyReNetv2(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth)

def InSPyReNetv2_SwinL(depth, pretrained):
    return InSPyReNetv2(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth)
