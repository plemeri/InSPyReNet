import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

class SPM(nn.Module):
    def __init__(self, model):
        super(SPM, self).__init__()
        self.model = model
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
            patch_size = sample['patch_size']
            stride = sample['stride']
        else:
            raise TypeError('input must be dict for SPM')
        
        x, shape = patch(x, patch_size, stride)
        b, c, h, w = shape
        out = self.model(x)
        out, _ = unpatch(out, (b, 1, h, w), patch_size, stride)
        return out