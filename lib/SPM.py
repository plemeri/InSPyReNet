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
        x = sample['image']
        patch_size = sample['patch_size']
        stride = sample['stride']
        
        x, shape = patch(x, patch_size, stride)
        b, c, h, w = shape
        out = self.model(x)
        out['pred'], _ = unpatch(out['pred'], (b, 1, h, w), patch_size, stride)

        pred = torch.sigmoid(out['pred'])
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = out['loss']
        return sample
