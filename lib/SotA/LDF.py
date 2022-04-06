## this code has been brought from https://github.com/weijun88/LDF ##

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinB

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
    def __init__(self, pretrained=False):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

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

    def initialize(self):
        if self.pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'), strict=False)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d   = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d   = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d   = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d   = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        return (out4b, out3b, out2b, out1b), (out4d, out3d, out2d, out1d)

    def initialize(self):
        weight_init(self)


class LDF(nn.Module):
    def __init__(self, depth, pretrained=False, **kwargs):
        super(LDF, self).__init__()
        self.bkbone   = ResNet()
        self.conv5b   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder  = Encoder()
        self.decoderb = Decoder()
        self.decoderd = Decoder()
        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        # self.initialize()

    # def resize(self, x):
    #     b, _, h, w = x.shape
    #     h = h if h % 32 == 0 else (h // 32) * 32
    #     w = w if w % 32 == 0 else (w // 32) * 32
    #     return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
    


    def forward(self, x, shape=None):
        # x = self.resize(x)
        out1, out2, out3, out4, out5 = self.bkbone(x['image'])
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        outb1 = self.decoderb([out5b, out4b, out3b, out2b])
        outd1 = self.decoderd([out5d, out4d, out3d, out2d])
        out1  = torch.cat([outb1, outd1], dim=1)
        outb2, outd2 = self.encoder(out1)
        outb2 = self.decoderb([out5b, out4b, out3b, out2b], outb2)
        outd2 = self.decoderd([out5d, out4d, out3d, out2d], outd2)
        out2  = torch.cat([outb2, outd2], dim=1)

        if shape is None:
            shape = x['image'].size()[2:]
        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outb1), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outd1), size=shape, mode='bilinear')

        out2  = F.interpolate(self.linear(out2),   size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard(outd2), size=shape, mode='bilinear')
        # return outb1, outd1, out1, outb2, outd2, out2
        return {'pred': torch.sigmoid(out2)}

    def initialize(self):
        weight_init(self)
        

class LDF_Res2Net50(nn.Module):
    def __init__(self, depth, pretrained=False, **kwargs):
        super(LDF_Res2Net50, self).__init__()
        self.bkbone   = res2net50_v1b_26w_4s(pretrained)
        self.conv5b   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder  = Encoder()
        self.decoderb = Decoder()
        self.decoderd = Decoder()
        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        # self.initialize()

    # def resize(self, x):
    #     b, _, h, w = x.shape
    #     h = h if h % 32 == 0 else (h // 32) * 32
    #     w = w if w % 32 == 0 else (w // 32) * 32
    #     return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
    


    def forward(self, x, shape=None):
        # x = self.resize(x)
        out1, out2, out3, out4, out5 = self.bkbone(x['image'])
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        outb1 = self.decoderb([out5b, out4b, out3b, out2b])
        outd1 = self.decoderd([out5d, out4d, out3d, out2d])
        out1  = torch.cat([outb1, outd1], dim=1)
        outb2, outd2 = self.encoder(out1)
        outb2 = self.decoderb([out5b, out4b, out3b, out2b], outb2)
        outd2 = self.decoderd([out5d, out4d, out3d, out2d], outd2)
        out2  = torch.cat([outb2, outd2], dim=1)

        if shape is None:
            shape = x['image'].size()[2:]
        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outb1), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outd1), size=shape, mode='bilinear')

        out2  = F.interpolate(self.linear(out2),   size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard(outd2), size=shape, mode='bilinear')
        # return outb1, outd1, out1, outb2, outd2, out2
        return {'pred': torch.sigmoid(out2)}

    def initialize(self):
        weight_init(self)
        
class LDF_SwinB(nn.Module):
    def __init__(self, depth, pretrained=False, **kwargs):
        super(LDF_SwinB, self).__init__()
        self.bkbone   = SwinB(pretrained)
        self.conv5b   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder  = Encoder()
        self.decoderb = Decoder()
        self.decoderd = Decoder()
        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        # self.initialize()

    # def resize(self, x):
    #     b, _, h, w = x.shape
    #     h = h if h % 32 == 0 else (h // 32) * 32
    #     w = w if w % 32 == 0 else (w // 32) * 32
    #     return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
    


    def forward(self, x, shape=None):
        # x = self.resize(x)
        out1, out2, out3, out4, out5 = self.bkbone(x['image'])
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        outb1 = self.decoderb([out5b, out4b, out3b, out2b])
        outd1 = self.decoderd([out5d, out4d, out3d, out2d])
        out1  = torch.cat([outb1, outd1], dim=1)
        outb2, outd2 = self.encoder(out1)
        outb2 = self.decoderb([out5b, out4b, out3b, out2b], outb2)
        outd2 = self.decoderd([out5d, out4d, out3d, out2d], outd2)
        out2  = torch.cat([outb2, outd2], dim=1)

        if shape is None:
            shape = x['image'].size()[2:]
        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outb1), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outd1), size=shape, mode='bilinear')

        out2  = F.interpolate(self.linear(out2),   size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard(outd2), size=shape, mode='bilinear')
        # return outb1, outd1, out1, outb2, outd2, out2
        return {'pred': torch.sigmoid(out2)}

    def initialize(self):
        weight_init(self)