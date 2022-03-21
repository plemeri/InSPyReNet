import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.ResNet import resnet50
from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinT, SwinS, SwinB, SwinL

from lib.InSPyReNet import InSPyReNet

class GLoSPyRe(InSPyReNet):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(GLoSPyRe, self).__init__(backbone, in_channels, depth, base_size, **kwargs)
        
    def forward(self, sample):
        x = sample['image']
        B, _, H, W = x.shape

        sample['image'] = self.res(x, self.base_size)
        gout = super(GLoSPyRe, self).forward(sample)
        sample['gpred'] = gout['gaussian'][-1]
        sample['image'] = x
        lout = super(GLoSPyRe, self).forward(sample)

        sample['pred'] = lout['pred']
        sample['loss'] = lout['loss'] + gout['loss']
        sample['gaussian'] = lout['gaussian']
        sample['laplacian'] = lout['laplacian']
        return sample
    
def GLoSPyRe_ResNet50(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(resnet50(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size)
    
def GLoSPyRe_Res2Net50(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def GLoSPyRe_Res2Net101(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def GLoSPyRe_SwinS(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)

def GLoSPyRe_SwinT(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth, base_size, **kwargs)
    
def GLoSPyRe_SwinB(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)

def GLoSPyRe_SwinL(depth, pretrained, base_size, **kwargs):
    return GLoSPyRe(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth, base_size, **kwargs)