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
        
        # for i, d in enumerate(out['gaussian']):
        #     d = torch.abs(.5 - torch.sigmoid(d)) * 2
        #     out['gaussian'][d < .5] = float('inf')
        
        od3, od2, od1, od0 = out['gaussian']
        op2, op1, op0 = out['laplacian']
        
        
        x, shape = patch(x, self.patch_size, self.stride)
        b, c, h, w = shape
        pout = self.model({'image': x})
        
        pd3, pd2, pd1, pd0 = pout['gaussian']
        pp2, pp1, pp0 = pout['laplacian']
        
        # d3  = unpatch(pd3, (b, 1, h // 8, w // 8), 
        #                   self.patch_size // 8, 
        #                   self.stride // 8)
        d3 = self.res(od3, (h // 8, w // 8))
        p2  = unpatch(pp2, (b, 1, h // 4, w // 4), 
                         self.patch_size // 4, 
                         self.stride // 4)
        d2 = self.model.pyr.rec(d3, p2)
        
        
        p1  = unpatch(pp1, (b, 1, h // 2, w // 2), 
                         self.patch_size // 2, 
                         self.stride // 2)
        
        d1 = self.model.pyr.rec(d2.detach(), p1)
        
        p0 = unpatch(pp0, (b, 1, h     , w     ), 
                        self.patch_size, 
                        self.stride)
        
        d0 =  self.model.pyr.rec(d1.detach(), p0)
        
        if type(sample) == dict:
            return {'pred': torch.sigmoid(d0), 
                    'loss': 0, 
                    'gaussian': [d3, d2, d1, d0], 
                    'laplacian': [p2, p1, p0]}
        
        else:
            return torch.sigmoid(d0)