import cv2
import torch.nn as nn
import torch.nn.functional as F

from kornia.morphology import dilation, erosion

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinT, SwinS, SwinB, SwinL

from lib.InSPyReNet import InSPyReNet

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
        
class GLoSS(InSPyReNet):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(GLoSS, self).__init__(backbone, in_channels, depth, base_size, **kwargs)
        self.transition0 = Transition(17)
        self.transition1 = Transition(9)
        self.transition2 = Transition(5)
        
    def cuda(self):
        super(GLoSS, self).cuda()
        self.transition0.cuda()
        self.transition1.cuda()
        self.transition2.cuda()
        return self

    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape

        # Global Saliency Pyramid & Reconstruction)
        sample['image'] = self.res(x, self.base_size)
        gout = super(GLoSS, self).forward(sample)
        gd3, gd2, gd1, gd0 = gout['gaussian']
        gp2, gp1, gp0 = gout['laplacian']
            
        # Local Saliency Pyramid
        sample['image'] = x
        lout = super(GLoSS, self).forward(sample)
        ld3, ld2, ld1, ld0 = lout['gaussian']
        lp2, lp1, lp0 = lout['laplacian']
        
        # Local Saliency Pyramid Reconstruction
        d3 = self.ret(gd0, ld3) 
        
        t2 = self.ret(self.transition2(d3), lp2)
        p2 = t2 * lp2
        d2 = self.pyr.rec(d3, p2)
        
        t1 = self.ret(self.transition1(d2), lp1)
        p1 = t1 * lp1
        d1 = self.pyr.rec(d2, p1)
        
        t0 = self.ret(self.transition0(d1), lp0)
        p0 = t0 * lp0
        d0 = self.pyr.rec(d1, p0)

        pred = torch.sigmoid(d0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['gaussian'] = [d3, d2, d1, d0]
        sample['laplacian'] = [p2, p1, p0]
        return sample
    
    
def GLoSS_Res2Net50(depth, pretrained, base_size, **kwargs):
    return GLoSS(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def GLoSS_Res2Net101(depth, pretrained, base_size, **kwargs):
    return GLoSS(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def GLoSS_SwinS(depth, pretrained, base_size, **kwargs):
    return GLoSS(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def GLoSS_SwinT(depth, pretrained, base_size, **kwargs):
    return GLoSS(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def GLoSS_SwinB(depth, pretrained, base_size, **kwargs):
    return GLoSS(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def GLoSS_SwinL(depth, pretrained, base_size, **kwargs):
    return GLoSS(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)