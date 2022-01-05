import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

class simple_context(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(simple_context, self).__init__()
        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = conv(in_channel, out_channel, 3, dilation=3)
        self.branch2 = conv(in_channel, out_channel, 3, dilation=5)
        self.branch3 = conv(in_channel, out_channel, 3, dilation=7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3, relu=True)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat([x0, x1, x2, x3], dim=1)
        x_cat = self.conv_cat(x_cat)

        x_cat = x_cat + self.conv_res(x)
        return x_cat

class PAA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size, stage_size=None):
        super(PAA_kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = self_attn(out_channel, 'h', stage_size)
        self.Wattn = self_attn(out_channel, 'w', stage_size)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x

class PAA_e(nn.Module):
    def __init__(self, in_channel, out_channel, base_size=None, stage=None):
        super(PAA_e, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = base_size // (2 ** stage)
        else:
            self.stage_size = None

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3, self.stage_size)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5, self.stage_size)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7, self.stage_size)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x