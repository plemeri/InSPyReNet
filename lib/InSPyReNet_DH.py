import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *
from lib.InSPyReNet import *

class InSPyReNet_DH(nn.Module):
    def __init__(self, model, pretrained, in_channels, depth=64):
        super(InSPyReNet_DH, self).__init__()
        self.model = model(depth, pretrained)
        self.model_dh = model(depth, pretrained)
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.pyr = Pyr(7, 1)
        
        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self.model = self.model.cuda()
        self.model_dh = self.model_dh.cuda()
        self = super(InSPyReNet_DH, self).cuda()
        return self
        
        
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
            dh = sample['depth']
        else:
            x, dh = sample
        
        B, C, H, W = x.shape
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            out = self.model({'image': x, 'gt': sample['gt']})
            out_dh = self.model_dh({'image': dh, 'gt': sample['gt']})
        else:
            out = self.model({'image': x})
            out_dh = self.model_dh({'image': dh})
            
        xd3, xd2, xd1, xd0 = out['gaussian']
        xp2, xp1, xp0 = out['laplacian']

        dd3, dd2, dd1, dd0 = out_dh['gaussian']
        dp2, dp1, dp0 = out_dh['laplacian']
        
        d3 = (xd3 + dd3) / 2
        p2 = (xp2 + dp2) / 2
        d2 = self.pyr.rec(d3.detach(), p2)
        
        p1 = (xp1 + dp1) / 2
        d1 = self.pyr.rec(d2.detach(), p1)
        
        p0 = (xp0 + dp0) / 2
        d0 =  self.pyr.rec(d1.detach(), p0)
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
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
            loss += out['loss'] + out_dh['loss']
            loss /= 3
        
        else:
            loss = 0
        
        if type(sample) == dict:
            return {'pred': xd0, 
                    'loss': loss, 
                    'gaussian': [d3, d2, d1, d0], 
                    'laplacian': [p2, p1, p0]}
        
        else:
            return d0
        
        
def InSPyReNet_DH_Res2Net50(depth, pretrained):
    return InSPyReNet_DH(InSPyReNet_Res2Net50, pretrained, [64, 256, 512, 1024, 2048], depth)

def InSPyReNet_DH_Res2Net101(depth, pretrained):
    return InSPyReNet_DH(InSPyReNet_Res2Net101, pretrained, [64, 256, 512, 1024, 2048], depth)

def InSPyReNet_DH_SwinS(depth, pretrained):
    return InSPyReNet_DH(InSPyReNet_SwinS, pretrained, [96, 96, 192, 384, 768], depth)

def InSPyReNet_DH_SwinT(depth, pretrained):
    return InSPyReNet_DH(InSPyReNet_SwinT, pretrained, [96, 96, 192, 384, 768], depth)
    
def InSPyReNet_DH_SwinB(depth, pretrained):
    return InSPyReNet_DH(InSPyReNet_SwinB, pretrained, [128, 128, 256, 512, 1024], depth)

def InSPyReNet_DH_SwinL(depth, pretrained):
    return InSPyReNet_DH(InSPyReNet_SwinL, pretrained, [192, 192, 384, 768, 1536], depth)