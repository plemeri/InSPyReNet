import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kornia.morphology import dilation, erosion

from lib.optim import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinT, SwinS, SwinB, SwinL
from lib.backbones.FFCResNet import ffc_resnet50

class Transition:
    def __init__(self, k=3):
        self.kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).float()
        
    def cuda(self):
        self.kernel = self.kernel.cuda()
        return self
        
    def __call__(self, x):
        x = torch.sigmoid(x)
        dx = dilation(x, self.kernel)
        ex = erosion(x, self.kernel)
        
        return ((dx - ex) > .5).float()

class InSPyReNetV2(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(InSPyReNetV2, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        self.decoder = simple_decoder(self.in_channels[4], self.depth)

        self.attention1 = SIOC(self.in_channels[0], 1, depth=self.depth, base_size=self.base_size, stage=0)
        self.attention2 = SIOC(self.in_channels[1], 1, depth=self.depth, base_size=self.base_size, stage=1)
        self.attention3 = SIOC(self.in_channels[2], 1, depth=self.depth, base_size=self.base_size, stage=2)
        self.attention4 = SIOC(self.in_channels[3], 1, depth=self.depth, base_size=self.base_size, stage=3)

        self.loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + focal_tversky_loss_with_logits(x, y, alpha=0.2, beta=0.8, gamma=2, reduction='mean')
        self.image_pyramidamidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.image_pyramid = ImagePyramid(7, 1)
        
        self.transition1 = Transition(31)
        self.transition2 = Transition(17)
        self.transition3 = Transition(9)
        self.transition4 = Transition(5)
        
    def cuda(self):
        self.image_pyramid.cuda()
        self.transition1.cuda()
        self.transition2.cuda()
        self.transition3.cuda()
        self.transition4.cuda()
        self = super(InSPyReNetV2, self).cuda()
        return self
    
    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        d5 = self.decoder(x5) #32

        p4 = self.attention4(x4, d5.detach())
        d4 = self.image_pyramid.reconstruct(d5.detach(), p4) #16

        p3 = self.attention3(x3, d4.detach())
        d3 = self.image_pyramid.reconstruct(d4.detach(), p3) #8

        p2 = self.attention2(x2, d3.detach())
        d2 = self.image_pyramid.reconstruct(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        p1 = self.attention1(x1, d2.detach())
        d1 = self.image_pyramid.reconstruct(d2.detach(), p1) #4
        
        d0 = self.res(d1, (H, W))
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            y1 = self.image_pyramid.down(y)
            y2 = self.image_pyramid.down(y1)
            y3 = self.image_pyramid.down(y2)
            y4 = self.image_pyramid.down(y3)
            y5 = self.image_pyramid.down(y4)

            ploss =  self.image_pyramidamidal_consistency_loss_fn(self.des(d5, (H, W)), self.des(self.image_pyramid.down(d4), (H, W)).detach()) * 0.0001
            ploss += self.image_pyramidamidal_consistency_loss_fn(self.des(d4, (H, W)), self.des(self.image_pyramid.down(d3), (H, W)).detach()) * 0.0001
            ploss += self.image_pyramidamidal_consistency_loss_fn(self.des(d3, (H, W)), self.des(self.image_pyramid.down(d2), (H, W)).detach()) * 0.0001
            ploss += self.image_pyramidamidal_consistency_loss_fn(self.des(d2, (H, W)), self.des(self.image_pyramid.down(d1), (H, W)).detach()) * 0.0001
            
            closs =   self.loss_fn(self.des(d5, (H, W)), self.des(y5, (H, W)))
            closs +=  self.loss_fn(self.des(d4, (H, W)), self.des(y4, (H, W)))
            closs +=  self.loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
            closs +=  self.loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
            closs +=  self.loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
            
            closs += self.loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
            
            loss = ploss + closs

        else:
            loss = 0
            
        pred = torch.sigmoid(d0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        sample['gaussian'] = [d5, d4, d3, d2, d1]
        sample['laplacian'] = [p4, p3, p2, p1]
        return sample
    
def InSPyReNetV2_FFCResNet50(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(ffc_resnet50(pretrained=pretrained, ratio=0.25), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)
    
def InSPyReNetV2_Res2Net50(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReNetV2_Res2Net101(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReNetV2_SwinS(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def InSPyReNetV2_SwinT(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def InSPyReNetV2_SwinB(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def InSPyReNetV2_SwinL(depth, pretrained, base_size, **kwargs):
    return InSPyReNetV2(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)

# class InSPyReNetV2D(InSPyReNetV2):
#     def __init__(self, backbone, in_channels, depth=64, base_size=384, **kwargs):
#         super(InSPyReNetV2D, self).__init__(backbone, in_channels, depth, base_size, **kwargs)
#         self.reduce = conv(4, 3, 3)
        
#     def forward(self, sample):
#         x = torch.cat([sample['image'], sample['depth']], dim=1)
#         x = self.reduce(x)
        
#         sample['image'] = x
#         return super(InSPyReNetV2D, self).forward(sample)
    

# def InSPyReNetV2D_Res2Net50(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetV2D(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def InSPyReNetV2D_Res2Net101(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetV2D(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def InSPyReNetV2D_SwinS(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetV2D(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

# def InSPyReNetV2D_SwinT(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetV2D(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
# def InSPyReNetV2D_SwinB(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetV2D(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

# def InSPyReNetV2D_SwinL(depth, pretrained, base_size, **kwargs):
#     return InSPyReNetV2D(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)