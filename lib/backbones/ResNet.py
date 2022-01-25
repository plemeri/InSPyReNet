## this code has been brought from https://github.com/weijun88/LDF ##

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], layers=[3, 4, 6, 3], strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1]):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer(channels[0], layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2   = self.make_layer(channels[1], layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3   = self.make_layer(channels[2], layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4   = self.make_layer(channels[3], layers[3], stride=strides[3], dilation=dilations[3])
        

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5
    
            
def resnet50(pretrained=True, **kwargs):
    model = ResNet(channels=[64, 128, 256, 512], layers=[3, 4, 6, 3], strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1])
    if pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(channels=[64, 128, 256, 512], layers=[3, 4, 23, 3], strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1])
    if pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'), strict=False)
    return model