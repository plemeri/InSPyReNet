import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

class InSPyReNet_PM(nn.Module):
    def __init__(self, model, patch_size, stride):
        super(InSPyReNet_PM, self).__init__()
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.truncate = 1.5
        
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
        else:
            x = sample
        
        out = self.model({'image': self.res(x, (self.patch_size, self.patch_size))})
        od3, od2, od1, od0 = out['gaussian']
        op2, op1, op0 = out['laplacian']
        
        
        x, shape = patch(x, self.patch_size, self.stride)
        b, c, h, w = shape
        pout = self.model({'image': x})
        
        pd3, pd2, pd1, pd0 = pout['gaussian']
        pp2, pp1, pp0 = pout['laplacian']
        
        d3, i3  = unpatch(pd3, (b, 1, h // 8, w // 8), self.patch_size // 8, self.stride // 8, guide=self.res(od3 - self.truncate, (h // 8, w // 8)))
        
        _, i2  = unpatch(pd2, (b, 1, h // 4, w // 4), self.patch_size // 4, self.stride // 4, guide=self.res(od2 - self.truncate, (h // 4, w // 4)))
        p2, _  = unpatch(pp2, (b, 1, h // 4, w // 4), self.patch_size // 4, self.stride // 4, indice_map = F.pixel_shuffle(torch.cat([i3] * 4, dim=1), 2), guide=self.res(op2, (h // 4, w // 4)))
        d2 = self.model.pyr.rec(d3.detach(), p2)
        
        _, i1  = unpatch(pd1, (b, 1, h // 2, w // 2), self.patch_size // 2, self.stride // 2, guide=self.res(od1 - self.truncate, (h // 2, w // 2)))
        p1, _  = unpatch(pp1, (b, 1, h // 2, w // 2), self.patch_size // 2, self.stride // 2, indice_map = F.pixel_shuffle(torch.cat([i2] * 4, dim=1), 2), guide=self.res(op1, (h // 2, w // 2)))
        d1 = self.model.pyr.rec(d2.detach(), p1)
        
        p0, _  = unpatch(pp0, (b, 1, h     , w     ), self.patch_size, self.stride, indice_map = F.pixel_shuffle(torch.cat([i1] * 4, dim=1), 2), guide=self.res(op0, (h, w)))
        d0 =  self.model.pyr.rec(d1.detach(), p0)
        
        if type(sample) == dict:
            return {'pred': d0, 
                    'loss': 0, 
                    'gaussian': [d3, d2, d1, d0], 
                    'laplacian': [p2, p1, p0]}
        
        else:
            return d0