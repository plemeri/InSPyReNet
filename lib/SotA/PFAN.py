## this code has been brought from https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection ##

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

vgg_conv1_2 = vgg_conv2_2 = vgg_conv3_3 = vgg_conv4_3 = vgg_conv5_3 = None

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size-1)//2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels//2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels//2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels//2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels//2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels//4)
        self.linear_2 = nn.Linear(self.in_channels//4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        
        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats, ca_act_reg

def conv_1_2_hook(module, input, output):
    global vgg_conv1_2
    vgg_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global vgg_conv2_2
    vgg_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global vgg_conv4_3
    vgg_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global vgg_conv5_3
    vgg_conv5_3 = output
    return None


class CPFE(nn.Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(CPFE, self).__init__()

        self.dil_rates = [3, 5, 7]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == 'conv5_3':
            self.in_channels = 512
        elif feature_layer == 'conv4_3':
            self.in_channels = 512
        elif feature_layer == 'conv3_3':
            self.in_channels = 256

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats


class PFAN(nn.Module):
    def __init__(self, depth, pretrained=False):
        super(PFAN, self).__init__()

        # Load the [partial] VGG-16 model
        self.vgg16 = models.vgg16(pretrained=pretrained).features

        # Extract and register intermediate features of VGG-16
        self.vgg16[3].register_forward_hook(conv_1_2_hook)
        self.vgg16[8].register_forward_hook(conv_2_2_hook)
        self.vgg16[15].register_forward_hook(conv_3_3_hook)
        self.vgg16[22].register_forward_hook(conv_4_3_hook)
        self.vgg16[29].register_forward_hook(conv_5_3_hook)

        # Initialize layers for high level (hl) feature (conv3_3, conv4_3, conv5_3) processing
        self.cpfe_conv3_3 = CPFE(feature_layer='conv3_3')
        self.cpfe_conv4_3 = CPFE(feature_layer='conv4_3')
        self.cpfe_conv5_3 = CPFE(feature_layer='conv5_3')

        self.cha_att = ChannelwiseAttention(in_channels=384)  # in_channels = 3 x (32 x 4)

        self.hl_conv1 = nn.Conv2d(384, 64, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(64)

        # Initialize layers for low level (ll) feature (conv1_2 and conv2_2) processing
        self.ll_conv_1 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.ll_bn_1 = nn.BatchNorm2d(64)
        self.ll_conv_2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(64)
        self.ll_conv_3 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_3 = nn.BatchNorm2d(64)

        self.spa_att = SpatialAttention(in_channels=64)

        # Initialize layers for fused features (ff) processing
        self.ff_conv_1 = nn.Conv2d(128, 1, (3, 3), padding=1)

    def forward(self, input_):
        global vgg_conv1_2, vgg_conv2_2, vgg_conv3_3, vgg_conv4_3, vgg_conv5_3

        # Pass input_ through vgg16 to generate intermediate features
        self.vgg16(input_)
        # print(vgg_conv1_2.size())
        # print(vgg_conv2_2.size())
        # print(vgg_conv3_3.size())
        # print(vgg_conv4_3.size())
        # print(vgg_conv5_3.size())

        # Process high level features
        conv3_cpfe_feats = self.cpfe_conv3_3(vgg_conv3_3)
        conv4_cpfe_feats = self.cpfe_conv4_3(vgg_conv4_3)
        conv5_cpfe_feats = self.cpfe_conv5_3(vgg_conv5_3)

        conv4_cpfe_feats = F.interpolate(conv4_cpfe_feats, scale_factor=2, mode='bilinear', align_corners=True)
        conv5_cpfe_feats = F.interpolate(conv5_cpfe_feats, scale_factor=4, mode='bilinear', align_corners=True)

        conv_345_feats = torch.cat((conv3_cpfe_feats, conv4_cpfe_feats, conv5_cpfe_feats), dim=1)

        conv_345_ca, ca_act_reg = self.cha_att(conv_345_feats)
        conv_345_feats = torch.mul(conv_345_feats, conv_345_ca)

        conv_345_feats = self.hl_conv1(conv_345_feats)
        conv_345_feats = F.relu(self.hl_bn1(conv_345_feats))
        conv_345_feats = F.interpolate(conv_345_feats, scale_factor=4, mode='bilinear', align_corners=True)

        # Process low level features
        conv1_feats = self.ll_conv_1(vgg_conv1_2)
        conv1_feats = F.relu(self.ll_bn_1(conv1_feats))
        conv2_feats = self.ll_conv_2(vgg_conv2_2)
        conv2_feats = F.relu(self.ll_bn_2(conv2_feats))

        conv2_feats = F.interpolate(conv2_feats, scale_factor=2, mode='bilinear', align_corners=True)
        conv_12_feats = torch.cat((conv1_feats, conv2_feats), dim=1)
        conv_12_feats = self.ll_conv_3(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_3(conv_12_feats))

        conv_12_sa = self.spa_att(conv_345_feats)
        conv_12_feats = torch.mul(conv_12_feats, conv_12_sa)

        # Fused features
        fused_feats = torch.cat((conv_12_feats, conv_345_feats), dim=1)
        fused_feats = self.ff_conv_1(fused_feats)

        return fused_feats #, ca_act_reg