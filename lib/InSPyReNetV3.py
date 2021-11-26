import torch.nn as nn
import torch.nn.functional as F

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *
from lib.InSPyReNet import *

class InSPyReNetV3(nn.Module):
    def __init__(self, backbone, in_channels, depth=64):
        super(InSPyReNetV3, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        
        self.RGBModules = nn.ModuleDict()
        self.DepthModules = nn.ModuleDict()
        
        for mdict in [self.RGBModules, self.DepthModules]:
            mdict.context1 = PAA_e(self.in_channels[0], self.depth)
            mdict.context2 = PAA_e(self.in_channels[1], self.depth)
            mdict.context3 = PAA_e(self.in_channels[2], self.depth)
            mdict.context4 = PAA_e(self.in_channels[3], self.depth)
            mdict.context5 = PAA_e(self.in_channels[4], self.depth)

            mdict.decoder = PAA_d(self.depth)

            mdict.attention0 = ASCA(self.depth    , depth, lmap_in=True)
            mdict.attention1 = ASCA(self.depth * 2, depth, lmap_in=True)
            mdict.attention2 = ASCA(self.depth * 2, depth)
        
        self.decoder = PAA_d(self.depth * 2)

        self.attention0 = ASCA(self.depth * 2, depth, lmap_in=True)
        self.attention1 = ASCA(self.depth * 4, depth, lmap_in=True)
        self.attention2 = ASCA(self.depth * 4, depth)
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.pyr = Pyr(7, 1)
        
        self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.pyramidal_consistency_loss_fn = nn.L1Loss()
        
    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(InSPyReNetV3, self).cuda()
        return self
    
    def forward_inspyre(self, mdict, x, y=None):
        B, _, H, W = x.shape
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = mdict.context1(x1) #4
        x2 = mdict.context2(x2) #4
        x3 = mdict.context3(x3) #8
        x4 = mdict.context4(x4) #16
        x5 = mdict.context5(x5) #32

        f3, d3 = mdict.decoder(x5, x4, x3) #16

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, p2 = mdict.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        d2 = self.pyr.rec(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = mdict.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach()) #2
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H, W))
        _, p0 = mdict.attention0(f1, d1.detach(), p1.detach()) #2
        d0 = self.pyr.rec(d1.detach(), p0) #2
        
        return {'backbone_feats': [x5, x4, x3, x2, x1],
                'feats': [f3, f2, f1],
                'laplacian': [p2, p1, p0],
                'gaussian': [d3, d2, d1, d0]}
        
        # if y is not None:
        #     y1 = self.pyr.down(y)
        #     y2 = self.pyr.down(y1)
        #     y3 = self.pyr.down(y2)

        #     ploss =  self.pyramidal_consistency_loss_fn(self.des(d3, (H, W)), self.des(self.pyr.down(d2), (H, W)).detach()) * 0.0001
        #     ploss += self.pyramidal_consistency_loss_fn(self.des(d2, (H, W)), self.des(self.pyr.down(d1), (H, W)).detach()) * 0.0001
        #     ploss += self.pyramidal_consistency_loss_fn(self.des(d1, (H, W)), self.des(self.pyr.down(d0), (H, W)).detach()) * 0.0001
            
        #     closs =  self.loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
        #     closs += self.loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
        #     closs += self.loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
        #     closs += self.loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
            
        #     loss = ploss + closs
        # else:
        #     loss = 0
        
        # return {'backbone_feats': [x5, x4, x3, x2, x1],
        #         'feats': [f3, f2, f1],
        #         'laplacian': [p2, p1, p0],
        #         'gaussian': [d3, d2, d1, d0],
        #         'loss': loss}
        
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
            dh = sample['depth']
        else:
            x, dh = sample
            
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
        else:
            y = None
            
        B, _, H, W = x.shape
        
        out = self.forward_inspyre(self.RGBModules, x, y)
        out_depth = self.forward_inspyre(self.DepthModules, dh, y)
        
        ox5, ox4, ox3, ox2, ox1 = out['backbone_feats']
        of3, of2, of1 = out['feats']
        op2, op1, op0 = out['laplacian']
        od3, od2, od1, od0 = out['gaussian']
        
        dx5, dx4, dx3, dx2, dx1 = out_depth['backbone_feats']
        df3, df2, df1 = out_depth['feats']
        dp2, dp1, dp0 = out_depth['laplacian']
        dd3, dd2, dd1, dd0 = out_depth['gaussian']
        
        x5, x4, x3, x2, x1 = torch.cat([ox5, dx5], dim=1), torch.cat([ox4, dx4], dim=1), torch.cat([ox3, dx3], dim=1), torch.cat([ox2, dx2], dim=1), torch.cat([ox1, dx1], dim=1)
        f3, f2, f1 = torch.cat([of3, df3], dim=1), torch.cat([of2, df2], dim=1), torch.cat([of1, df1], dim=1)
        
        f3, d3 = self.decoder(x5, x4, x3) #16
        d3 = (d3 + od3 + dd3) / 3

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        p2 = (p2 + op2 + dp2) / 3
        d2 = self.pyr.rec(d3.detach(), p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach()) #2
        p1 = (p1 + op1 + dp1) / 3
        d1 = self.pyr.rec(d2.detach(), p1) #2
        
        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1.detach(), p1.detach()) #2
        p0 = (p0 + op0 + dp0) / 3
        d0 = self.pyr.rec(d1.detach(), p0) #2
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            
            y1 = self.pyr.down(y)
            y2 = self.pyr.down(y1)
            y3 = self.pyr.down(y2)

            ploss =  self.pyramidal_consistency_loss_fn(self.des(d3, (H, W)), self.des(self.pyr.down(d2), (H, W)).detach()) * 0.0001
            ploss += self.pyramidal_consistency_loss_fn(self.des(d2, (H, W)), self.des(self.pyr.down(d1), (H, W)).detach()) * 0.0001
            ploss += self.pyramidal_consistency_loss_fn(self.des(d1, (H, W)), self.des(self.pyr.down(d0), (H, W)).detach()) * 0.0001
            
            closs =  self.loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
            closs += self.loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
            closs += self.loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
            closs += self.loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
            
            loss = ploss + closs

        else:
            loss = 0
            
        # loss += out['loss'] + out_depth['loss']

        if type(sample) == dict:
            return {'pred': d0, 
                    'loss': loss, 
                    'gaussian': [d3, d2, d1, d0], 
                    'laplacian': [p2, p1, p0]}
        
        else:
            return d0
        
        
def InSPyReNetV3_Res2Net50(depth, pretrained):
    return InSPyReNet(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetV3_Res2Net101(depth, pretrained):
    return InSPyReNet(res2net101_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth)

def InSPyReNetV3_SwinS(depth, pretrained):
    return InSPyReNet(SwinS(pretrained=pretrained), [96, 96, 192, 384, 768], depth)

def InSPyReNetV3_SwinT(depth, pretrained):
    return InSPyReNet(SwinT(pretrained=pretrained), [96, 96, 192, 384, 768], depth)
    
def InSPyReNetV3_SwinB(depth, pretrained):
    return InSPyReNet(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth)

def InSPyReNetV3_SwinL(depth, pretrained):
    return InSPyReNet(SwinL(pretrained=pretrained), [192, 192, 384, 768, 1536], depth)