import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

class PPM(nn.Module):
    def __init__(self, model):
        super(PPM, self).__init__()
        self.model = model
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self.model = self.model.cuda()
        self = super(PPM, self).cuda()
        return self
        
    def forward(self, sample):
        x = sample['image']
        patch_size = sample['patch_size']
        stride = sample['stride']
        
        x, shape = patch(x, patch_size, stride)
        b, c, h, w = shape
        out = self.model({'image': x})
        
        pd3, pd2, pd1, pd0 = out['gaussian']
        pp2, pp1, pp0 = out['laplacian']
        
        d3, i3  = unpatch(pd3, (b, 1, h // 8, w // 8), patch_size // 8, stride // 8)
        
        _, i2  = unpatch(pd2, (b, 1, h // 4, w // 4), patch_size // 4, stride // 4)
        p2, _  = unpatch(pp2, (b, 1, h // 4, w // 4), patch_size // 4, stride // 4, indice_map = F.pixel_shuffle(torch.cat([i3] * 4, dim=1), 2))
        d2 = self.pyr.rec(d3.detach(), p2)
        
        _, i1  = unpatch(pd1, (b, 1, h // 2, w // 2), patch_size // 2, stride // 2)
        p1, _  = unpatch(pp1, (b, 1, h // 2, w // 2), patch_size // 2, stride // 2, indice_map = F.pixel_shuffle(torch.cat([i2] * 4, dim=1), 2))
        d1 = self.pyr.rec(d2.detach(), p1)
        
        p0, _  = unpatch(pp0, (b, 1, h     , w     ), patch_size, stride, indice_map = F.pixel_shuffle(torch.cat([i1] * 4, dim=1), 2))
        d0 =  self.pyr.rec(d1.detach(), p0)
        
        sample['pred'] = d0
        sample['loss'] = out['loss']
        sample['gaussian'] = [d3, d2, d1, d0]
        sample['laplacian'] = [p2, p1, p0]
        return sample