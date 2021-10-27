import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from utils.utils import *

class InSPyReNet_Grid(nn.Module):
    def __init__(self, model, patch_size):
        super(InSPyReNet_Grid, self).__init__()
        self.model = model
        self.patch_size = patch_size
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
    def forward(self, sample):
        sample['image'], shape = patch(sample['image'], self.patch_size)
        b, c, h, w = shape
        out = self.model(sample)
        
        pd3, pd2, pd1, pd0 = out['gaussian']
        pp2, pp1, pp0 = out['laplacian']
        
        d3, i3 = unpatch(pd3, (b, 1, h // 8, w // 8), self.patch_size // 8)
        
        _, i2  = unpatch(pd2, (b, 1, h // 4, w // 4), self.patch_size // 4)
        p2, _  = unpatch(pp2, (b, 1, h // 4, w // 4), self.patch_size // 4, indice_map = i2)
        d2 = self.model.pyr.rec(d3.detach(), p2)
        
        _, i1  = unpatch(pd1, (b, 1, h // 2, w // 2), self.patch_size // 2)
        p1, _  = unpatch(pp1, (b, 1, h // 2, w // 2), self.patch_size // 2, indice_map = i1)
        d1 = self.model.pyr.rec(d2.detach(), p1)
        
        _, i0  = unpatch(pd0, (b, 1, h     , w     ), self.patch_size)
        p0, _  = unpatch(pp0, (b, 1, h     , w     ), self.patch_size,      indice_map = i0)
        d0 =  self.model.pyr.rec(d1.detach(), p0)
        
        return {'pred': d0, 'debug': [d3, d2, d1, d0]}