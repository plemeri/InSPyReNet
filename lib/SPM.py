import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

class SPM(nn.Module):
    def __init__(self, model, patch_size, stride):
        super(SPM, self).__init__()
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
        else:
            x = sample
        
        x, shape = patch(x, self.patch_size, self.stride)
        b, c, h, w = shape
        out = self.model(x)
        out, _ = unpatch(out, (b, 1, h, w), self.patch_size, self.stride)
        return out