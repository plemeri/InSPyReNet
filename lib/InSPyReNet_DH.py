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
    def __init__(self, in_channels, depth):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        self.decoder = PAA_d(self.depth)

        self.attention0 = ASCA(self.depth    , depth, lmap_in=True)
        self.attention1 = ASCA(self.depth * 2, depth, lmap_in=True)
        self.attention2 = ASCA(self.depth * 2, depth)
        
        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
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

class InSPyReNet_DH(nn.Module):
    def __init__(self, backbone, in_channels, depth=64):
        super(InSPyReNet_DH, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        
        self.rgb_encoder = Encoder(in_channels, depth)
        self.rgb_decoder = Decoder(in_channels, depth)
        
        self.d_encoder = Encoder(in_channels, depth)
        # self.rgbd_decoder = Decoder(in_channels, depth * 2)
        
        # self.alpha3 = nn.Sequential(conv(depth * 3, depth, 3, relu=True), conv(depth, 1, 3, bn=False))
        # self.alpha2 = nn.Sequential(conv(depth * 3, depth, 3, relu=True), conv(depth, 1, 3, bn=False))
        # self.alpha1 = nn.Sequential(conv(depth * 3, depth, 3, relu=True), conv(depth, 1, 3, bn=False))
        
        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self.rgb_decoder = self.rgb_decoder.cuda()
        self.rgbd_decoder = self.rgbd_decoder.cuda()
        self = super(InSPyReNet_DH, self).cuda()
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
        xs = self.rgb_encoder(xs)
        x_out = self.rgb_decoder(xs, x.shape, y)
        
        ds = self.backbone(dh)
        ds = self.d_encoder(ds)
        xds = [torch.cat([x_.detach(), d_], dim=1) for x_, d_ in zip(xs, ds)]
        xd_out = self.rgbd_decoder(xds, x.shape, y)
    
        xd3, xd2, xd1, xd0 = x_out['gaussian']
        xp2, xp1, xp0 = x_out['laplacian']
        xf3, xf2, xf1 = x_out['feats']
        
        xdd3, xdd2, xdd1, xdd0 = xd_out['gaussian']
        xdp2, xdp1, xdp0 = xd_out['laplacian']
        xdf3, xdf2, xdf1 = xd_out['feats']
    
        a3 = torch.sigmoid(self.alpha3(torch.cat([xf3.detach(), xdf3], dim=1)))
        d3 = xd3.detach() * a3 + xdd3 * (1 - a3)
        
        a3 = self.res(a3, (H // 4, W // 4))
        p2 = xp2.detach() * a3 + xdp2 * (1 - a3)
        d2 = self.pyr.rec(d3.detach(), p2) #4
        
        a2 = torch.sigmoid(self.alpha3(torch.cat([xf2.detach(), xdf2], dim=1)))
        d2 = xd2.detach() * a2 + xdd2 * (1 - a2)
        
        a2 = self.res(a2, (H // 2, W // 2))
        p1 = xp1.detach() * a2 + xdp1 * (1 - a2)
        d1 = self.pyr.rec(d2.detach(), p1) #4
        
        a1 = torch.sigmoid(self.alpha3(torch.cat([xf1.detach(), xdf1], dim=1)))
        d1 = xd1.detach() * a1 + xdd1 * (1 - a1)
        
        a1 = self.res(a1, (H, W))
        p0 = xp0.detach() * a1 + xdp0 * (1 - a1)
        d0 = self.pyr.rec(d1.detach(), p0) #4
        
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
        loss += x_out['loss'] + xd_out['loss']

        if type(sample) == dict:
            return {'pred': d0, 
                    'loss': loss, 
                    'gaussian': [d3, d2, d1, d0], 
                    'laplacian': [p2, p1, p0],
                    'rgb_gaussian': x_out['gaussian'],
                    'rgb_laplacian': x_out['laplacian'],
                    'rgbd_gaussian': xd_out['gaussian'],
                    'rgbd_laplacian': xd_out['laplacian']}
        
        else:
            return d0
    
    
def InSPyReNet_DH_Res2Net50(depth, pretrained):
    return InSPyReNet_DH(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNet_DH_Res2Net101(depth, pretrained):
    return InSPyReNet_DH(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNet_DH_SwinS(depth, pretrained):
    return InSPyReNet_DH(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth)

def InSPyReNet_DH_SwinT(depth, pretrained):
    return InSPyReNet_DH(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth)
    
def InSPyReNet_DH_SwinB(depth, pretrained):
    return InSPyReNet_DH(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth)

def InSPyReNet_DH_SwinL(depth, pretrained):
    return InSPyReNet_DH(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth)