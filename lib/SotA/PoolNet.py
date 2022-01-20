## this code has been brought from https://github.com/backseason/PoolNet ##

import torch.nn.functional as F
from torch.autograd import Variable

import torch.nn as nn
import math
import torch
import numpy as np
affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x


class ResNet_locate(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_locate,self).__init__()
        self.resnet = ResNet(block, layers)
        self.in_planes = 512
        self.out_planes = [512, 256, 256, 128]

        self.ppms_pre = nn.Conv2d(2048, self.in_planes, 1, 1, bias=False)
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

def resnet50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model


config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 256, 'edgeinfoc':[48,128], 'block': [[512, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16], True]}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True]}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

class BlockLayer(nn.Module):
    def __init__(self, k_in, k_out_list):
        super(BlockLayer, self).__init__()
        up_in1, up_mid1, up_in2, up_mid2, up_out = [], [], [], [], []

        for k in k_out_list:
            up_in1.append(nn.Conv2d(k_in, k_in//4, 1, 1, bias=False))
            up_mid1.append(nn.Sequential(nn.Conv2d(k_in//4, k_in//4, 3, 1, 1, bias=False), nn.Conv2d(k_in//4, k_in, 1, 1, bias=False)))
            up_in2.append(nn.Conv2d(k_in, k_in//4, 1, 1, bias=False))
            up_mid2.append(nn.Sequential(nn.Conv2d(k_in//4, k_in//4, 3, 1, 1, bias=False), nn.Conv2d(k_in//4, k_in, 1, 1, bias=False)))
            up_out.append(nn.Conv2d(k_in, k, 1, 1, bias=False))

        self.block_in1 = nn.ModuleList(up_in1)
        self.block_in2 = nn.ModuleList(up_in2)
        self.block_mid1 = nn.ModuleList(up_mid1)
        self.block_mid2 = nn.ModuleList(up_mid2)
        self.block_out = nn.ModuleList(up_out)
        self.relu = nn.ReLU()

    def forward(self, x, mode=0):
        x_tmp = self.relu(x + self.block_mid1[mode](self.block_in1[mode](x)))
        # x_tmp = self.block_mid2[mode](self.block_in2[mode](self.relu(x + x_tmp)))
        x_tmp = self.relu(x_tmp + self.block_mid2[mode](self.block_in2[mode](x_tmp)))
        x_tmp = self.block_out[mode](x_tmp)

        return x_tmp

class EdgeInfoLayerC(nn.Module):
    def __init__(self, k_in, k_out):
        super(EdgeInfoLayerC, self).__init__()
        self.trans = nn.Sequential(nn.Conv2d(k_in, k_in, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_in, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True))

    def forward(self, x, x_size):
        tmp_x = []
        for i_x in x:
            tmp_x.append(F.interpolate(i_x, x_size[2:], mode='bilinear', align_corners=True))
        x = self.trans(torch.cat(tmp_x, dim=1))
        return x

class FuseLayer1(nn.Module):
    def __init__(self, list_k, deep_sup):
        super(FuseLayer1, self).__init__()
        up = []
        for i in range(len(list_k)):
            up.append(nn.Conv2d(list_k[i], 1, 1, 1))
        self.trans = nn.ModuleList(up)
        self.fuse = nn.Conv2d(len(list_k), 1, 1, 1)
        self.deep_sup = deep_sup

    def forward(self, list_x, x_size):
        up_x = []
        for i, i_x in enumerate(list_x):
            up_x.append(F.interpolate(self.trans[i](i_x), x_size[2:], mode='bilinear', align_corners=True))
        out_fuse = self.fuse(torch.cat(up_x, dim = 1))
        if self.deep_sup:
            out_all = []
            for up_i in up_x:
                out_all.append(up_i)
            return [out_fuse, out_all]
        else:
            return [out_fuse]

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 3, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

def extra_layer(base_model_cfg, base):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers = [], [], [], [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for k in config['block']:
        block_layers += [BlockLayer(k[0], k[1])]

    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    fuse_layers = FuseLayer1(config['fuse'][0], config['fuse'][1])
    edgeinfo_layers = EdgeInfoLayerC(config['edgeinfoc'][0], config['edgeinfoc'][1])
    score_layers = ScoreLayer(config['score'])

    return base, convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers


class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.block = nn.ModuleList(block_layers)
        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.fuse = fuse_layers
        self.edgeinfo = edgeinfo_layers
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

    def forward(self, x, mode=1):
        x_size = x.size()
        conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        edge_merge = []
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        edge_merge.append(merge)
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])
            edge_merge.append(merge)

        if mode == 0:
            edge_merge = [self.block[i](kk) for i, kk in enumerate(edge_merge)]
            merge = self.fuse(edge_merge, x_size)
        elif mode == 1:
            merge = self.deep_pool[-1](merge)
            edge_merge = [self.block[i](kk).detach() for i, kk in enumerate(edge_merge)]
            edge_merge = self.edgeinfo(edge_merge, merge.size())
            merge = self.score(torch.cat([merge, edge_merge], dim=1), x_size)
        return merge

# def build_model(base_model_cfg='vgg'):
#     if base_model_cfg == 'vgg':
#         return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
#     elif base_model_cfg == 'resnet':
#         return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))
    
def PoolNet_ResNet(depth, pretrained=False, **kwargs):
    net = PoolNet('resnet', *extra_layer('resnet', resnet50_locate()))
    # if pretrained is True:
        
    return net

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()