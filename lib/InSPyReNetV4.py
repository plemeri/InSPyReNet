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

class Encoder(nn.Module):
    def __init__(self, in_channels, depth):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        
        self.context1 = PAA_e(self.in_channels[0], self.depth)
        self.context2 = PAA_e(self.in_channels[1], self.depth)
        self.context3 = PAA_e(self.in_channels[2], self.depth)
        self.context4 = PAA_e(self.in_channels[3], self.depth)
        self.context5 = PAA_e(self.in_channels[4], self.depth)

    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(Encoder, self).cuda()
        return self

    def forward(self, xs):
        x1, x2, x3, x4, x5 = xs
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32
        
        return x1, x2, x3, x4, x5
    
class Decoder(nn.Module):
    def __init__(self, in_channels, depth, loss_fn):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        self.decoder = PAA_d(self.depth)

        self.attention0 = ASCA(self.depth    , depth, lmap_in=True)
        self.attention1 = ASCA(self.depth * 2, depth, lmap_in=True)
        self.attention2 = ASCA(self.depth * 2, depth)
        
        self.loss_fn = loss_fn
        self.pyramidal_consistency_loss_fn = nn.L1Loss()
        
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
    
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(Decoder, self).cuda()
        return self
        
    def forward(self, xs, shape, y=None):
        x1, x2, x3, x4, x5 = xs
        B, _, H, W  = shape

        f3, d3 = self.decoder(x5, x4, x3) #16

        f2, p2 = self.attention2(torch.cat([x2, self.res(f3, (H // 4,  W // 4 ))], dim=1), d3.detach())
        d2 = self.pyr.rec(d3.detach(), p2) #4

        f1, p1 = self.attention1(torch.cat([self.res(x1, (H // 2, W // 2)), self.res(f2, (H // 2, W // 2))], dim=1), d2.detach(), p2.detach()) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        _, p0 = self.attention0(self.res(f1, (H, W)), d1.detach(), p1.detach()) #2
        d0 = self.pyr.rec(d1.detach(), p0) #2
        
        if y is not None:
            
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
        else:
            loss = 0
        
        return {'gaussian': [d3, d2, d1, d0], 
                'laplacian': [p2, p1, p0],
                'feats': [f3, f2, f1],
                'loss': loss}

class InSPyReNetV4(nn.Module):
    def __init__(self, backbone, in_channels, depth=64):
        super(InSPyReNetV4, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        
        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.depth_loss_fn = nn.MSELoss()
        self.pyramidal_consistency_loss_fn = nn.L1Loss()
        
        self.rgb_encoder = Encoder(in_channels, depth)
        self.rgb_decoder = Decoder(in_channels, depth, self.loss_fn)
        
        self.d_encoder = Encoder(in_channels, depth)
        self.d_decoder = Decoder(in_channels, depth, self.depth_loss_fn)
        
        self.context1 = PAA_e(self.depth * 2, self.depth)
        self.context2 = PAA_e(self.depth * 2, self.depth)
        self.context3 = PAA_e(self.depth * 2, self.depth)
        self.context4 = PAA_e(self.depth * 2, self.depth)
        self.context5 = PAA_e(self.depth * 2, self.depth)
        
        self.rgbd_decoder = Decoder(in_channels, depth, self.loss_fn)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self.rgb_decoder = self.rgb_decoder.cuda()
        self.d_decoder = self.d_decoder.cuda()
        self.rgbd_decoder = self.rgbd_decoder.cuda()
        self = super(InSPyReNetV4, self).cuda()
        return self
    
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
            dh = sample['depth']
        else:
            x, dh = sample
            
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
        else:
            y = None
            
        B, _, H, W = x.shape
        xs = self.backbone(x)
        rs = self.rgb_encoder(xs)
        r_out = self.rgb_decoder(rs, x.shape, y)
        
        ds = self.d_encoder(xs)
        d_out = self.d_decoder(ds, x.shape, dh)
        
        rd_out = self.rgbd_decoder([con(torch.cat([r, d], dim=1)) for r, d, con in zip(rs, ds, [self.context1, self.context2, self.context3, self.context4, self.context5])], x.shape, y)
    
        _, _, _, r0 = r_out['gaussian']
        _, _, _, d0 = d_out['gaussian']
        _, _, _, rd0 = rd_out['gaussian']

        loss = r_out['loss'] + d_out['loss'] + rd_out['loss']

        if type(sample) == dict:
            return {'pred': rd0,
                    'loss': loss,
                    'rgb_gaussian': r_out['gaussian'],
                    'rgb_laplacian': r_out['laplacian'],
                    'd_gaussian': d_out['gaussian'],
                    'd_laplacian': d_out['laplacian']}
        
        else:
            return rd0
    
    
def InSPyReNetV4_Res2Net50(depth, pretrained):
    return InSPyReNetV4(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetV4_Res2Net101(depth, pretrained):
    return InSPyReNetV4(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetV4_SwinS(depth, pretrained):
    return InSPyReNetV4(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth)

def InSPyReNetV4_SwinT(depth, pretrained):
    return InSPyReNetV4(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth)
    
def InSPyReNetV4_SwinB(depth, pretrained):
    return InSPyReNetV4(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth)

def InSPyReNetV4_SwinL(depth, pretrained):
    return InSPyReNetV4(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth)