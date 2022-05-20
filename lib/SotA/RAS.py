import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinB
from lib.optim.losses import bce_loss_with_logits, iou_loss, iou_loss_with_logits

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
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
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

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
        return out2, out3, out4, out5

    def initialize(self):
        res50 = models.resnet50(pretrained=True)
        self.load_state_dict(res50.state_dict(), False)


class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.score = nn.Conv2d(out_channel*4, 1, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.bn(self.convert(x)))
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)

        return x

    def initialize(self):
        weight_init(self)


class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        a = torch.sigmoid(-y)
        x = self.relu(self.bn(self.convert(x)))
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        x = self.convs(x)
        # y = y + x

        return x

    def initialize(self):
        weight_init(self)

class RAS(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, **kwargs):
        super(RAS, self).__init__()
        # self.bkbone = ResNet()
        self.bkbone = backbone #SwinB(True)
        self.mscm = MSCM(in_channels[4], depth)
        self.ra2 = RA(in_channels[1], depth)
        self.ra3 = RA(in_channels[2], depth)
        self.ra4 = RA(in_channels[3], depth)

        self.loss_fn = lambda x, y: bce_loss_with_logits(x, y, 'mean') + iou_loss_with_logits(x, y, 'mean')

        # self.initialize()

    def forward(self, sample):
        x = sample['image']
        _, x2, x3, x4, x5 = self.bkbone(x)
        x_size = x.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        z5 = self.mscm(x5)

        z5_4 = F.interpolate(z5, x4_size, mode='bilinear', align_corners=True)
        y4 = self.ra4(x4, z5_4)
        
        z4 = z5_4 + y4
        z4_3 = F.interpolate(z4, x3_size, mode='bilinear', align_corners=True)
        y3 = self.ra3(x3, z4_3)

        z3 = z4_3 + y3
        z3_2 = F.interpolate(z3, x2_size, mode='bilinear', align_corners=True)	
        
        y2 = self.ra2(x2, z3_2)
        z2 = z3_2 + y2

        pred = torch.sigmoid(z2)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
                
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
        
            score5 = F.interpolate(z5, y.shape[-2:], mode='bilinear', align_corners=True)
            score4 = F.interpolate(z4, y.shape[-2:], mode='bilinear', align_corners=True)
            score3 = F.interpolate(z3, y.shape[-2:], mode='bilinear', align_corners=True)
            score2 = F.interpolate(z2, y.shape[-2:], mode='bilinear', align_corners=True)
            
            loss =  self.loss_fn(score2, y)
            loss += self.loss_fn(score3, y)
            loss += self.loss_fn(score4, y)
            loss += self.loss_fn(score5, y)

        else:
            loss = 0

        sample['pred'] = pred
        # sample['aux'] = [score3, score4, score5]
        sample['gaussian'] = z5, z4, z3, z2
        sample['laplacian'] = y4, y3, y2
        return sample

    # def initialize(self):
        # if self.cfg.snapshot:
            # self.load_state_dict(torch.load(self.cfg.snapshot))
        # else:
            # weight_init(self)
    
def RAS_Res2Net50(depth, pretrained, **kwargs):
    return RAS(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, **kwargs)
    
def RAS_SwinB(depth, pretrained, **kwargs):
    return RAS(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, **kwargs)

