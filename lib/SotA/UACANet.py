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

class UACANet(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(UACANet, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        self.context2 = PAA_e(512,  depth)
        self.context3 = PAA_e(1024, depth)
        self.context4 = PAA_e(2048, depth)

        self.decoder = PAA_d(depth)

        self.attention2 = UACA(depth * 2, depth)
        self.attention3 = UACA(depth * 2, depth)
        self.attention4 = UACA(depth * 2, depth)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        self.loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x, y, reduction='mean')


        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.image_pyramid = ImagePyramid(7, 1)
        
    def cuda(self):
        self.image_pyramid = self.image_pyramid.cuda()
        self = super(UACANet, self).cuda()
        return self
    
    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x3 = self.context2(x3)
        x4 = self.context3(x4)
        x5 = self.context4(x5)

        f5, d5 = self.decoder(x5, x4, x3)

        f4, p4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), d5)
        d4 = self.ret(d5, p4) + p4

        f3, p3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), d4)
        d3 = self.ret(d4, p3) + p3

        _, p2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), d3)
        d2 = self.ret(d3, p2) + p2
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            loss5 = self.loss_fn(self.res(d5, (H, W)), y)
            loss4 = self.loss_fn(self.res(d4, (H, W)), y)
            loss3 = self.loss_fn(self.res(d3, (H, W)), y)
            loss2 = self.loss_fn(self.res(d2, (H, W)), y)

            loss = loss2 + loss3 + loss4 + loss5
        else:
            loss = 0
            
        d0 = self.res(d2, (H, W))
        pred = torch.sigmoid(d0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        sample['gaussian'] = [d5, d4, d3, d2]
        sample['laplacian'] = [p4, p3, p2]
        return sample
    
def UACANet_FFCResNet50(depth, pretrained, base_size, **kwargs):
    return UACANet(ffc_resnet50(pretrained=pretrained, ratio=0.25), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)
    
def UACANet_Res2Net50(depth, pretrained, base_size, **kwargs):
    return UACANet(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def UACANet_Res2Net101(depth, pretrained, base_size, **kwargs):
    return UACANet(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def UACANet_SwinS(depth, pretrained, base_size, **kwargs):
    return UACANet(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def UACANet_SwinT(depth, pretrained, base_size, **kwargs):
    return UACANet(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def UACANet_SwinB(depth, pretrained, base_size, **kwargs):
    return UACANet(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def UACANet_SwinL(depth, pretrained, base_size, **kwargs):
    return UACANet(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)

# class UACANetD(UACANet):
#     def __init__(self, backbone, in_channels, depth=64, base_size=384, **kwargs):
#         super(UACANetD, self).__init__(backbone, in_channels, depth, base_size, **kwargs)
#         self.reduce = conv(4, 3, 3)
        
#     def forward(self, sample):
#         x = torch.cat([sample['image'], sample['depth']], dim=1)
#         x = self.reduce(x)
        
#         sample['image'] = x
#         return super(UACANetD, self).forward(sample)
    

# def UACANetD_Res2Net50(depth, pretrained, base_size, **kwargs):
#     return UACANetD(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def UACANetD_Res2Net101(depth, pretrained, base_size, **kwargs):
#     return UACANetD(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

# def UACANetD_SwinS(depth, pretrained, base_size, **kwargs):
#     return UACANetD(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

# def UACANetD_SwinT(depth, pretrained, base_size, **kwargs):
#     return UACANetD(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
# def UACANetD_SwinB(depth, pretrained, base_size, **kwargs):
#     return UACANetD(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

# def UACANetD_SwinL(depth, pretrained, base_size, **kwargs):
#     return UACANetD(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)