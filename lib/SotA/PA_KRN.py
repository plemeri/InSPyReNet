import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
                 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False],
                               [True, True, True, True, False]], 'score': 128}

affine_par = True

def fmax(data, axis=None, keepdims=None):
    if axis is None:
        return torch.max(data, keepdim=keepdims)[0]
    return torch.max(data, axis, keepdim=keepdims)[0]


def fsum(data, axis=None, keepdims=None):
    if axis is None:
        return torch.sum(data, keepdim=keepdims)
    return torch.sum(data, axis, keepdim=keepdims)


def fminimum(lhs, rhs, out=None):
    assert isinstance(lhs, torch.Tensor)
    if isinstance(rhs, torch.Tensor):
        return torch.min(lhs, rhs, out=out)
    if out is None:
        return lhs.clamp_max(rhs)
    return lhs.clamp(max=rhs, out=out)


broadcast_minimum = fminimum

cumsum = torch.cumsum
def fempty(shape, ctx=None):
    return torch.empty(shape, device=ctx)

def fget_ctx(data):
    return data.device


def ftile(data, reps):
    return data.repeat(reps)


def freshape(data, shape):
    return data.view(shape)

class AttSamplerGrid:
    def __init__(self, scale=1.0, dense=4, iters=5):
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def forward(self, data, attx, atty):#data[1, 1, 224, 224]  attx[1, 224, 1] 
        # attx: (N, W, 1)
        # atty: (N, H, 1)
        N, _, in_size, in_sizey = data.shape#N大概是batch_size. in_size是图片的宽度224
        att_size = attx.shape[1]#的到图片的宽度
        att_sizey = atty.shape[1]

        out_size = int(in_size * self.scale)#输出的size和输入的size一样大
        out_sizey = int(in_sizey * self.scale)#输出的size和输入的size一样大
        #print('out_sizey',out_sizey)
        
        #threshold应该是 方大缩小的界限
        threshold = float(self.scale * self.dense * in_size) / att_size
        
        #print('threshold',threshold)
        #attention的尺寸根据输入和输出的尺寸改变
        attx = attx * out_size
        atty = atty * out_sizey
        #print('attx',attx)
        for j in range(self.iters):
            max_attx = fmax(attx, 1, keepdims=True)  # (N, 1, 1)
            #print('max_attx',max_attx)
            max_atty = fmax(atty, 1, keepdims=True)  # (N, 1, 1)
            #print('max_atty',max_atty)
            if j == 0:
                threshold = fminimum(fminimum(
                    max_attx, max_atty), threshold)  # (N, 1, 1)
            else:
                threshold = fminimum(max_attx, max_atty)
            #print(j)
            #print(threshold)
            #print('attx',attx)
            
            broadcast_minimum(threshold, attx, out=attx)
            #print('attx',attx)
            broadcast_minimum(threshold, atty, out=atty)
            sum_x = fsum(attx, 1, keepdims=True)  # (N, 1, 1)
            sum_y = fsum(atty, 1, keepdims=True)  # (N, 1, 1)
            deltax = (out_size - sum_x) / att_size
            deltay = (out_sizey - sum_y) / att_sizey
            # compensate the drop value
            attx += deltax
            atty += deltay
        '''
        it is the same as the original implemenation.
        the first element is 1.
        '''
        attx[:, 0] = 1
        #print(attx)
        atty[:, 0] = 1
        
        #产生逆变换函数的过程
        attxi = cumsum(attx, 1)#新的attention坐标300
        #print('attxi',attxi)
        attyi = cumsum(atty, 1)

        stepx = attxi[:, -1] / out_size
        stepy = attyi[:, -1] / out_sizey #stepy tensor([[1.0034]])
        ctx = fget_ctx(stepx)
        
        #创建随机变量的过程
        index_x = fempty((N, out_sizey, 1), ctx=ctx)#-1,1 应该是
        index_y = fempty((N, out_size, 1), ctx=ctx)

        #应该是逆变换的过程（离散逆变换的过程，涉及插值的部分）
        # mobula.func.map_step(N, attxi, index_y, stepx, att_size, out_size)
        # mobula.func.map_step(N, attyi, index_x, stepy, att_sizey, out_sizey)
        #GG = F.tile(freshape(index_x, (N, 1, out_sizey)), (1, out_size, 1))
        #MM = ftile(index_y, (1, 1, out_sizey))
        #print('GG',GG)
        #print('GG',GG.shape)
        #print('MM',MM)
        #print('GG',MM.shape)
        return ftile(freshape(index_x, (N, 1, out_sizey)), (1, out_size, 1)),\
            ftile(index_y, (N, 1, out_sizey))

    def backward(self, dy_x, dy_y):
        return [0, 0, 0]

    def infer_shape(self, in_shape):
        #in_shape [torch.Size([1, 1, 342, 400]), torch.Size([1, 342, 1]), torch.Size([1, 400, 1])]
        dshape = in_shape[0]
        out_size = int(dshape[2] * self.scale)
        #dshape1 = in_shape[1]
        out_size1 = int(dshape[3] * self.scale)
        #print('out_size1',out_size1.shape)
        oshape = (dshape[0], out_size, out_size1)
        return in_shape, [oshape, oshape]

class AttSampler(torch.nn.Module):
    def __init__(self, scale=1.0, dense=4, iters=5):
        super(AttSampler, self).__init__()
        self.scale = scale
        self.dense = dense
        self.iters = iters
        
        self.grid = AttSamplerGrid(scale=self.scale,
                            dense=self.dense,
                            iters=self.iters)

    def forward(self, data, attx, atty):
        grid = self.grid.forward(data.detach(), attx.detach(), atty.detach())
                            
        grid = torch.stack(grid, dim=3)
        #print('data_data',data.shape)
        #print('grid_grid',grid.shape)
        #print('grid',grid)
        return F.grid_sample(data, grid),grid

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

class DeepPoolLayer_first(nn.Module):
    def __init__(self, k, k_out, need_x2,
                 need_fuse):  # (config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])
        super(DeepPoolLayer_first, self).__init__()
        self.pools_sizes = [2, 2, 2]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None):  # (merge, conv2merge[k+1], infos[k])
        x_size = x.size()
        resl = x
        y = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](y))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(resl, x2))
        return resl


class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg_krn
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, score_layers = [], []
    convert_layers = ConvertLayer(config['convert'])
    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, score_layers


class KRN(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(KRN, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base  # 基本网络是一样的

        #self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

        # 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]]
        self.DeepPool_solid1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_solid2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_solid3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_solid4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_solid5 = DeepPoolLayer_first(128, 128, False, False)

        self.DeepPool_contour1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_contour2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_contour3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_contour4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_contour5 = DeepPoolLayer_first(128, 128, False, False)

        self.relu = nn.ReLU()
        self.conv_reduce1 = nn.Conv2d(512, 128, 1, 1, 1, bias=False)
        self.conv_reduce2 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)
        self.conv_reduce3 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)

        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)

        self.score_solid1 = ScoreLayer(512)
        self.score_contour1 = ScoreLayer(512)

        self.score_solid2 = ScoreLayer(256)
        self.score_contour2 = ScoreLayer(256)

        self.score_solid3 = ScoreLayer(256)
        self.score_contour3 = ScoreLayer(256)

        self.score_solid4 = ScoreLayer(128)
        self.score_contour4 = ScoreLayer(128)

        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)
        self.score_sum_out = ScoreLayer(128)

        self.conv_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.conv_add1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add3 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add4 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_sum_out = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)  #
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
        out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
        out_merge_solid1 = F.sigmoid(out_merge_solid1)
        fea_reduce1 = self.conv_reduce1(merge_solid1)
        fea_reduce1 = self.relu(fea_reduce1)

        merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
        out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
        out_merge_solid2 = F.sigmoid(out_merge_solid2)
        fea_reduce2 = self.conv_reduce2(merge_solid2)
        fea_reduce2 = self.relu(fea_reduce2)

        merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
        out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
        out_merge_solid3 = F.sigmoid(out_merge_solid3)
        fea_reduce3 = self.conv_reduce3(merge_solid3)
        fea_reduce3 = self.relu(fea_reduce3)

        merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
        out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
        out_merge_solid4 = F.sigmoid(out_merge_solid4)
        fea_reduce4 = merge_solid4

        merge_solid5 = self.DeepPool_solid5(merge_solid4)
        merge_solid = self.score_solid(merge_solid5, x_size)
        merge_solid = F.sigmoid(merge_solid)

        fea_reduce1 = F.interpolate(fea_reduce1, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce2 = F.interpolate(fea_reduce2, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce3 = F.interpolate(fea_reduce3, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce4 = F.interpolate(fea_reduce4, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_add1 = torch.add(merge_solid5, fea_reduce1)
        fea_add2 = torch.add(merge_solid5, fea_reduce2)
        fea_add3 = torch.add(merge_solid5, fea_reduce3)
        fea_add4 = torch.add(merge_solid5, fea_reduce4)
        fea_add1 = self.conv_add1(fea_add1)
        fea_add1 = self.relu(fea_add1)
        fea_add2 = self.conv_add2(fea_add2)
        fea_add2 = self.relu(fea_add2)
        fea_add3 = self.conv_add3(fea_add3)
        fea_add3 = self.relu(fea_add3)
        fea_add4 = self.conv_add4(fea_add4)
        fea_add4 = self.relu(fea_add4)
        feasum_out = torch.cat((fea_add1, fea_add2, fea_add3, fea_add4), 1)
        feasum_out = self.conv_sum_out(feasum_out)
        feasum_out = self.score_sum_out(feasum_out, x_size)
        feasum_out = F.sigmoid(feasum_out)

        return feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4


class KRN_edge(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(KRN_edge, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base  # 基本网络是一样的

        #self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

        # 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]]
        self.DeepPool_solid1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_solid2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_solid3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_solid4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_solid5 = DeepPoolLayer_first(128, 128, False, False)

        self.DeepPool_contour1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_contour2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_contour3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_contour4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_contour5 = DeepPoolLayer_first(128, 128, False, False)

        self.relu = nn.ReLU()
        self.conv_reduce1 = nn.Conv2d(512, 128, 1, 1, 1, bias=False)
        self.conv_reduce2 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)
        self.conv_reduce3 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)

        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)

        self.score_solid1 = ScoreLayer(512)
        self.score_contour1 = ScoreLayer(512)

        self.score_solid2 = ScoreLayer(256)
        self.score_contour2 = ScoreLayer(256)

        self.score_solid3 = ScoreLayer(256)
        self.score_contour3 = ScoreLayer(256)

        self.score_solid4 = ScoreLayer(128)
        self.score_contour4 = ScoreLayer(128)

        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)
        self.score_sum_out = ScoreLayer(128)

        self.conv_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.conv_add1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add3 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add4 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_sum_out = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)  #
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        merge_contour1 = self.conv_1(conv2merge[1])
        merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
        out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
        out_merge_contour1 = self.score_contour1(merge_contour1, x_size)
        out_merge_solid1 = F.sigmoid(out_merge_solid1)
        out_merge_contour1 = F.sigmoid(out_merge_contour1)
        # merge_contour1, merge_solid1 = self.fuse1(merge_contour1, merge_solid1)
        fea_reduce1 = self.conv_reduce1(merge_solid1)
        fea_reduce1 = self.relu(fea_reduce1)

        merge_contour2 = self.conv_2(conv2merge[2])
        merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
        out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
        out_merge_contour2 = self.score_contour2(merge_contour2, x_size)
        out_merge_solid2 = F.sigmoid(out_merge_solid2)
        out_merge_contour2 = F.sigmoid(out_merge_contour2)
        # merge_contour2, merge_solid2 = self.fuse2(merge_contour2, merge_solid2)
        fea_reduce2 = self.conv_reduce2(merge_solid2)
        fea_reduce2 = self.relu(fea_reduce2)

        merge_contour3 = self.conv_3(conv2merge[3])
        merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
        out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
        out_merge_contour3 = self.score_contour3(merge_contour3, x_size)
        out_merge_solid3 = F.sigmoid(out_merge_solid3)
        out_merge_contour3 = F.sigmoid(out_merge_contour3)
        # merge_contour3, merge_solid3 = self.fuse3(merge_contour3, merge_solid3)
        fea_reduce3 = self.conv_reduce3(merge_solid3)
        fea_reduce3 = self.relu(fea_reduce3)

        merge_contour4 = self.conv_4(conv2merge[4])
        merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
        out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
        out_merge_contour4 = self.score_contour4(merge_contour4, x_size)
        out_merge_solid4 = F.sigmoid(out_merge_solid4)
        out_merge_contour4 = F.sigmoid(out_merge_contour4)
        # merge_contour4, merge_solid4 = self.fuse4(merge_contour4, merge_solid4)
        fea_reduce4 = merge_solid4

        merge_solid5 = self.DeepPool_solid5(merge_solid4)
        merge_solid = self.score_solid(merge_solid5, x_size)
        merge_solid = F.sigmoid(merge_solid)

        fea_reduce1 = F.interpolate(fea_reduce1, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce2 = F.interpolate(fea_reduce2, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce3 = F.interpolate(fea_reduce3, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce4 = F.interpolate(fea_reduce4, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_add1 = torch.add(merge_solid5, fea_reduce1)
        fea_add2 = torch.add(merge_solid5, fea_reduce2)
        fea_add3 = torch.add(merge_solid5, fea_reduce3)
        fea_add4 = torch.add(merge_solid5, fea_reduce4)
        fea_add1 = self.conv_add1(fea_add1)
        fea_add1 = self.relu(fea_add1)
        fea_add2 = self.conv_add2(fea_add2)
        fea_add2 = self.relu(fea_add2)
        fea_add3 = self.conv_add3(fea_add3)
        fea_add3 = self.relu(fea_add3)
        fea_add4 = self.conv_add4(fea_add4)
        fea_add4 = self.relu(fea_add4)
        feasum_out = torch.cat((fea_add1, fea_add2, fea_add3, fea_add4), 1)
        feasum_out = self.conv_sum_out(feasum_out)
        feasum_out = self.score_sum_out(feasum_out, x_size)
        feasum_out = F.sigmoid(feasum_out)

        return feasum_out, merge_solid, out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'resnet':
        return KRN(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
            
class PA_KRN(nn.Module):
    def __init__(self, depth, pretrained=False):
        super().__init__()
        self.net = KRN('resnet', *extra_layer('resnet', resnet50_locate()))
        self.net = self.net #.cuda()
        self.net.eval()
        # self.net.load_state_dict(torch.load(self.config.clm_model))

        self.net_hou = KRN_edge('resnet', *extra_layer('resnet', resnet50_locate()))
        self.net_hou = self.net_hou #.cuda()
        self.net_hou.eval()
        # self.net_hou.load_state_dict(torch.load(self.config.fsm_model))
        
    def forward(self, images):
        feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4 = self.net(
                    images)


        map_s = feasum_out

        map_sx = torch.unsqueeze(torch.max(map_s, 3)[0], dim=3)  # ([1, 400, 1])
        map_sx = torch.squeeze(map_sx, dim=1)
        map_sy = torch.unsqueeze(torch.max(map_s, 2)[0], dim=3)  # ([1, 342, 1])
        map_sy = torch.squeeze(map_sy, dim=1)
        sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
        sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
        map_sx /= sum_sx
        map_sy /= sum_sy

        semi_pred, grid = AttSampler(scale=1, dense=2)(images, map_sx, map_sy)
        mapsssss, grid5 = AttSampler(scale=1, dense=2)(map_s, map_sx, map_sy)
        # mapsssss,grid5 = AttSampler(scale=1, dense=2)(map_s, map_sx, map_sy)
        data_pred, merge_solid, out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4 = self.net_hou(
            semi_pred)

        ##################################restore##############################################
        x_index = grid[0, 1, :, 0]  # 400
        y_index = grid[0, :, 1, 1]  # 300

        new_data_size = tuple(data_pred.shape[1:4])
        new_data = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                device=images.device)  # 创建新的图
        new_data_final = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                        device=images.device)  # 创建新的图
        x_index = (x_index + 1) * new_data_size[2] / 2
        y_index = (y_index + 1) * new_data_size[1] / 2

        xl = 0
        grid_l = x_index[0]
        data_l = data_pred[:, :, :, 0]
        for num in range(1, len(x_index)):
            grid_r = x_index[num]
            xr = torch.ceil(grid_r) - 1
            xr = xr.int()
            data_r = data_pred[:, :, :, num]
            for h in range(xl + 1, xr + 1):
                print(h)
                if h == grid_r:
                    new_data[:, :, h] = data_r
                else:
                    new_data[:, :, h] = ((h - grid_l) * data_r / (grid_r - grid_l)) + (
                                (grid_r - h) * data_l / (grid_r - grid_l))
            xl = xr
            grid_l = grid_r
            data_l = data_r
        new_data[:, :, 0] = new_data[:, :, 1]
        try:
            for h in range(xr + 1, len(x_index)):
                new_data[:, :, h] = new_data[:, :, xr]
        except:
            print('h', h)
            print('xr', xr)

        yl = 0
        grid1_l = y_index[0]
        data1_l = new_data[:, 0, :]
        for num in range(1, len(y_index)):
            grid1_r = y_index[num]
            yr = torch.ceil(grid1_r) - 1
            yr = yr.int()
            data1_r = new_data[:, num, :]
            for h in range(yl + 1, yr + 1):
                if h == grid1_r:
                    new_data_final[:, h, :] = data1_r
                else:
                    new_data_final[:, h, :] = ((h - grid1_l) * data1_r / (grid1_r - grid1_l)) + (
                                (grid1_r - h) * data1_l / (grid1_r - grid1_l))
            yl = yr
            grid1_l = grid1_r
            data1_l = data1_r
        new_data_final[:, 0, :] = new_data_final[:, 1, :]
        try:
            for h in range(yr + 1, len(y_index)):
                new_data_final[:, h, :] = new_data_final[:, yr, :]
        except:
            print('h', h)
            print('yr', yr)
        preds = torch.unsqueeze(new_data_final, dim=1)

        pred = np.squeeze(preds).cpu().data.numpy()
        multi_fuse = 255 * pred