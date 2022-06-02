from email.mime import base
import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

# if indice_map is None:
#     out, ind = torch.max(out, dim=0)
# else:
#     ind = indice_map
#     out = torch.gather(out, 0, ind.unsqueeze(0)).squeeze(0)

class InSPyReNet_MS(nn.Module):
    def __init__(self, model, base_size=384, scales=[0.5]):
        super(InSPyReNet_MS, self).__init__()
        self.model = model
        self.scales = scales
        self.base_size = base_size
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.image_pyramid = self.model.pyr
        
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
        else:
            x = sample
            
        B, C, H, W = x.shape
        
        out_g = []
        out_l = []
        
        for scale in self.scales:
            if H * scale < self.base_size or W * scale < self.base_size:
                continue
            else:
                out = self.model({'image': self.res(x, (int(H * scale), int(W * scale)))})
                self.model.patch_size = (self.base_size * scale)
                out_g.append(out['gaussian'])
                out_l.append(out['laplacian'])
        
        d3 = torch.cat([self.res(i, (H // 8, W // 8)) for i in out_g[0]], dim=1)
        print(d3.shape)
        d3, i3 = torch.max(d3, dim=1, keepdim=True)
        
        # p2 = torch.gather(torch.cat([self.res(i, (H // 4, W // 4)) for i in out_l[0]], dim=0), 0, F.pixel_shuffle(torch.cat([i3] * 4, dim=1), 2))
        # d2, i2 = torch.max(torch.cat([self.res(i, (H // 4, W // 4)) for i in out_g[1]], dim=0), dim=0, keepdim=True)
        # d2 = self.image_pyramid.reconstruct(d3.detach(), p2)
        
        # p1 = torch.gather(torch.cat([self.res(i, (H // 2, W // 2)) for i in out_l[1]], dim=0), 0, F.pixel_shuffle(torch.cat([i2] * 4, dim=1), 2))
        # d1, i1 = torch.max(torch.cat([self.res(i, (H // 2, W // 2)) for i in out_g[2]], dim=0), dim=0, keepdim=True)
        # d1 = self.image_pyramid.reconstruct(d2.detach(), p1)
        
        # p0 = torch.gather(torch.cat([self.res(i,  (H, W)) for i in out_l[2]], dim=0), 0, F.pixel_shuffle(torch.cat([i1] * 4, dim=1), 2))
        # d0 = self.image_pyramid.reconstruct(d1.detach(), p0)
        
        # print(d3.shape, d2.shape, d1.shape, d0.shape)
        
        if type(sample) == dict:
            return {'pred': out['pred'], 
                    'loss': 0}
                    # 'gaussian': [d3, d2, d1, d0], 
                    # 'laplacian': [p2, p1, p0]}
        
        else:
            return d3