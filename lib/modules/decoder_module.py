import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

class PPD(nn.Module):
    def __init__(self, channel):
        super(PPD, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        self.conv_upsample1 = conv(channel, channel, 3)
        self.conv_upsample2 = conv(channel, channel, 3)
        self.conv_upsample3 = conv(channel, channel, 3)
        self.conv_upsample4 = conv(channel, channel, 3)
        self.conv_upsample5 = conv(2 * channel, 2 * channel, 3)

        self.conv_concat2 = conv(2 * channel, 2 * channel, 3)
        self.conv_concat3 = conv(3 * channel, 3 * channel, 3)
        self.conv4 = conv(3 * channel, 3 * channel, 3)
        self.conv5 = conv(3 * channel, 1, 1, bn=False, bias=True)

    def forward(self, f1, f2, f3):
        f1x2 = self.upsample(f1, f2.shape[-2:])
        f1x4 = self.upsample(f1, f3.shape[-2:])
        f2x2 = self.upsample(f2, f3.shape[-2:])

        f2_1 = self.conv_upsample1(f1x2) * f2
        f3_1 = self.conv_upsample2(f1x4) * self.conv_upsample3(f2x2) * f3

        f1_2 = self.conv_upsample4(f1x2)
        f2_2 = torch.cat([f2_1, f1_2], 1)
        f2_2 = self.conv_concat2(f2_2)

        f2_2x2 = self.upsample(f2_2, f3.shape[-2:])
        f2_2x2 = self.conv_upsample5(f2_2x2)

        f3_2 = torch.cat([f3_1, f2_2x2], 1)
        f3_2 = self.conv_concat3(f3_2)

        f3_2 = self.conv4(f3_2)
        out = self.conv5(f3_2)

        return f3_2, out

class simple_decoder(nn.Module):
    def __init__(self, channel):
        super(simple_decoder, self).__init__()
        self.conv1 = conv(channel * 3, channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        
    def forward(self, f1, f2, f3):
        f1 = self.upsample(f1, f3.shape[-2:])
        f2 = self.upsample(f2, f3.shape[-2:])
        f3 = torch.cat([f1, f2, f3], dim=1)

        f3 = self.conv1(f3)
        f3 = self.conv2(f3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        out = self.conv5(f3)

        return f3, out

class PAA_d(nn.Module):
    def __init__(self, channel):
        super(PAA_d, self).__init__()
        self.conv1 = conv(channel * 3 ,channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        
    def forward(self, f1, f2, f3):
        f1 = self.upsample(f1, f3.shape[-2:])
        f2 = self.upsample(f2, f3.shape[-2:])
        f3 = torch.cat([f1, f2, f3], dim=1)
        f3 = self.conv1(f3)

        Hf3 = self.Hattn(f3)
        Wf3 = self.Wattn(f3)

        f3 = self.conv2(Hf3 + Wf3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        out = self.conv5(f3)

        return f3, out

class PAA_d2(nn.Module):
    def __init__(self, channel):
        super(PAA_d2, self).__init__()
        self.conv1 = conv(channel * 2 ,channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        
    def forward(self, f2, f3):
        f2 = self.upsample(f2, f3.shape[-2:])
        f3 = torch.cat([f2, f3], dim=1)
        f3 = self.conv1(f3)

        Hf3 = self.Hattn(f3)
        Wf3 = self.Wattn(f3)

        f3 = self.conv2(Hf3 + Wf3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        out = self.conv5(f3)

        return f3, out
