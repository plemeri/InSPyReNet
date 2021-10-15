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
        sample['image'], shape = patch_max(sample['image'], self.patch_size)
        b, c, h, w = shape
        out = self.model(sample)
        
        dd3, pp2, pp1, pp = out['debug']
        d3 = stitch_avg_mod(dd3, (b, 1, h // 16, w // 16), self.patch_size // 16)
        p2 = stitch_avg_mod(pp2, (b, 1, h // 8, w // 8), self.patch_size // 8)
        p1 = stitch_avg_mod(pp1, (b, 1, h // 4, w // 4), self.patch_size // 4)
        p =  stitch_avg_mod(pp, (b, 1, h // 2, w // 2), self.patch_size // 2)
        
        d2 = self.model.inspyre.rec(d3.detach(), p2)
        d1 = self.model.inspyre.rec(d2.detach(), p1)
        d =  self.model.inspyre.rec(d1.detach(), p)
        d = self.res(d, (h, w))
        
        return {'pred': d, 'debug': [d3, d2, d1, out['pred']]}