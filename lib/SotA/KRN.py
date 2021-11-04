# import torch
# from torch import nn
# import torch.nn.functional as F

# config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
#                  'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False],
#                                [True, True, True, True, False]], 'score': 128}

# affine_par = True

# def max(data, axis=None, keepdims=None):
#     if axis is None:
#         return torch.max(data, keepdim=keepdims)[0]
#     return torch.max(data, axis, keepdim=keepdims)[0]


# def sum(data, axis=None, keepdims=None):
#     if axis is None:
#         return torch.sum(data, keepdim=keepdims)
#     return torch.sum(data, axis, keepdim=keepdims)


# def minimum(lhs, rhs, out=None):
#     assert isinstance(lhs, torch.Tensor)
#     if isinstance(rhs, torch.Tensor):
#         return torch.min(lhs, rhs, out=out)
#     if out is None:
#         return lhs.clamp_max(rhs)
#     return lhs.clamp(max=rhs, out=out)


# broadcast_minimum = minimum

# cumsum = torch.cumsum
# def empty(shape, ctx=None):
#     return torch.empty(shape, device=ctx)

# def get_ctx(data):
#     return data.device


# def tile(data, reps):
#     return data.repeat(reps)


# def reshape(data, shape):
#     return data.view(shape)

# torch._mobula_hack = mobula_hack_for_pytorch

# class AttSampler(torch.nn.Module):
#     def __init__(self, scale=1.0, dense=4, iters=5):
#         super(AttSampler, self).__init__()
#         self.scale = scale
#         self.dense = dense
#         self.iters = iters

#     def forward(self, data, attx, atty):
#         grid = mobula.op.AttSamplerGrid(data.detach(),
#                                         attx.detach(),
#                                         atty.detach(),
#                                         scale=self.scale,
#                                         dense=self.dense,
#                                         iters=self.iters)
#         grid = torch.stack(grid, dim=3)
#         #print('data_data',data.shape)
#         #print('grid_grid',grid.shape)
#         #print('grid',grid)
#         return F.grid_sample(data, grid),grid

# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
#         self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
#         for i in self.bn1.parameters():
#             i.requires_grad = False
#         padding = 1
#         if dilation_ == 2:
#             padding = 2
#         elif dilation_ == 4:
#             padding = 4
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
#                                padding=padding, bias=False, dilation = dilation_)
#         self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
#         for i in self.bn2.parameters():
#             i.requires_grad = False
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
#         for i in self.bn3.parameters():
#             i.requires_grad = False
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, layers):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
#         for i in self.bn1.parameters():
#             i.requires_grad = False
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
#             )
#         for i in downsample._modules['1'].parameters():
#             i.requires_grad = False
#         layers = []
#         layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes,dilation_=dilation__))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         tmp_x = []
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         tmp_x.append(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         tmp_x.append(x)
#         x = self.layer2(x)
#         tmp_x.append(x)
#         x = self.layer3(x)
#         tmp_x.append(x)
#         x = self.layer4(x)
#         tmp_x.append(x)

#         return tmp_x


# class ResNet_locate(nn.Module):
#     def __init__(self, block, layers):
#         super(ResNet_locate,self).__init__()
#         self.resnet = ResNet(block, layers)
#         self.in_planes = 512
#         self.out_planes = [512, 256, 256, 128]

#         self.ppms_pre = nn.Conv2d(2048, self.in_planes, 1, 1, bias=False)
#         ppms, infos = [], []
#         for ii in [1, 3, 5]:
#             ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
#         self.ppms = nn.ModuleList(ppms)

#         self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
#         for ii in self.out_planes:
#             infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
#         self.infos = nn.ModuleList(infos)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def load_pretrained_model(self, model):
#         self.resnet.load_state_dict(model, strict=False)

#     def forward(self, x):
#         x_size = x.size()[2:]
#         xs = self.resnet(x)

#         xs_1 = self.ppms_pre(xs[-1])
#         xls = [xs_1]
#         for k in range(len(self.ppms)):
#             xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
#         xls = self.ppm_cat(torch.cat(xls, dim=1))

#         infos = []
#         for k in range(len(self.infos)):
#             infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

#         return xs, infos

# def resnet50_locate():
#     model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
#     return model

# class ConvertLayer(nn.Module):
#     def __init__(self, list_k):
#         super(ConvertLayer, self).__init__()
#         up = []
#         for i in range(len(list_k[0])):
#             up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
#         self.convert0 = nn.ModuleList(up)

#     def forward(self, list_x):
#         resl = []
#         for i in range(len(list_x)):
#             resl.append(self.convert0[i](list_x[i]))
#         return resl

# class DeepPoolLayer_first(nn.Module):
#     def __init__(self, k, k_out, need_x2,
#                  need_fuse):  # (config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])
#         super(DeepPoolLayer_first, self).__init__()
#         self.pools_sizes = [2, 2, 2]
#         self.need_x2 = need_x2
#         self.need_fuse = need_fuse
#         pools, convs = [], []
#         for i in self.pools_sizes:
#             pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
#             convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
#         self.pools = nn.ModuleList(pools)
#         self.convs = nn.ModuleList(convs)
#         self.relu = nn.ReLU()
#         self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
#         if self.need_fuse:
#             self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

#     def forward(self, x, x2=None):  # (merge, conv2merge[k+1], infos[k])
#         x_size = x.size()
#         resl = x
#         y = x
#         for i in range(len(self.pools_sizes)):
#             y = self.convs[i](self.pools[i](y))
#             resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
#         resl = self.relu(resl)
#         if self.need_x2:
#             resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
#         resl = self.conv_sum(resl)
#         if self.need_fuse:
#             resl = self.conv_sum_c(torch.add(resl, x2))
#         return resl


# class ScoreLayer(nn.Module):
#     def __init__(self, k):
#         super(ScoreLayer, self).__init__()
#         self.score = nn.Conv2d(k, 1, 1, 1)

#     def forward(self, x, x_size=None):
#         x = self.score(x)
#         if x_size is not None:
#             x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
#         return x


# def extra_layer(base_model_cfg, vgg):
#     if base_model_cfg == 'vgg':
#         config = config_vgg_krn
#     elif base_model_cfg == 'resnet':
#         config = config_resnet
#     convert_layers, score_layers = [], []
#     convert_layers = ConvertLayer(config['convert'])
#     score_layers = ScoreLayer(config['score'])

#     return vgg, convert_layers, score_layers


# class KRN(nn.Module):
#     def __init__(self, base_model_cfg, base, convert_layers, score_layers):
#         super(KRN, self).__init__()
#         self.base_model_cfg = base_model_cfg
#         self.base = base  # 基本网络是一样的

#         #self.deep_pool = nn.ModuleList(deep_pool_layers)
#         self.score = score_layers
#         if self.base_model_cfg == 'resnet':
#             self.convert = convert_layers

#         # 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]]
#         self.DeepPool_solid1 = DeepPoolLayer_first(512, 512, False, True)
#         self.DeepPool_solid2 = DeepPoolLayer_first(512, 256, True, True)
#         self.DeepPool_solid3 = DeepPoolLayer_first(256, 256, True, True)
#         self.DeepPool_solid4 = DeepPoolLayer_first(256, 128, True, True)
#         self.DeepPool_solid5 = DeepPoolLayer_first(128, 128, False, False)

#         self.DeepPool_contour1 = DeepPoolLayer_first(512, 512, False, True)
#         self.DeepPool_contour2 = DeepPoolLayer_first(512, 256, True, True)
#         self.DeepPool_contour3 = DeepPoolLayer_first(256, 256, True, True)
#         self.DeepPool_contour4 = DeepPoolLayer_first(256, 128, True, True)
#         self.DeepPool_contour5 = DeepPoolLayer_first(128, 128, False, False)

#         self.relu = nn.ReLU()
#         self.conv_reduce1 = nn.Conv2d(512, 128, 1, 1, 1, bias=False)
#         self.conv_reduce2 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)
#         self.conv_reduce3 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)

#         self.score_solid = ScoreLayer(128)
#         self.score_contour = ScoreLayer(128)

#         self.score_solid1 = ScoreLayer(512)
#         self.score_contour1 = ScoreLayer(512)

#         self.score_solid2 = ScoreLayer(256)
#         self.score_contour2 = ScoreLayer(256)

#         self.score_solid3 = ScoreLayer(256)
#         self.score_contour3 = ScoreLayer(256)

#         self.score_solid4 = ScoreLayer(128)
#         self.score_contour4 = ScoreLayer(128)

#         self.score_solid = ScoreLayer(128)
#         self.score_contour = ScoreLayer(128)
#         self.score_sum_out = ScoreLayer(128)

#         self.conv_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
#         self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
#         self.conv_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
#         self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

#         self.conv_add1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_add2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_add3 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_add4 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_sum_out = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

#     def forward(self, x):
#         x_size = x.size()
#         conv2merge, infos = self.base(x)  #
#         if self.base_model_cfg == 'resnet':
#             conv2merge = self.convert(conv2merge)
#         conv2merge = conv2merge[::-1]

#         merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
#         out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
#         out_merge_solid1 = F.sigmoid(out_merge_solid1)
#         fea_reduce1 = self.conv_reduce1(merge_solid1)
#         fea_reduce1 = self.relu(fea_reduce1)

#         merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
#         out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
#         out_merge_solid2 = F.sigmoid(out_merge_solid2)
#         fea_reduce2 = self.conv_reduce2(merge_solid2)
#         fea_reduce2 = self.relu(fea_reduce2)

#         merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
#         out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
#         out_merge_solid3 = F.sigmoid(out_merge_solid3)
#         fea_reduce3 = self.conv_reduce3(merge_solid3)
#         fea_reduce3 = self.relu(fea_reduce3)

#         merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
#         out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
#         out_merge_solid4 = F.sigmoid(out_merge_solid4)
#         fea_reduce4 = merge_solid4

#         merge_solid5 = self.DeepPool_solid5(merge_solid4)
#         merge_solid = self.score_solid(merge_solid5, x_size)
#         merge_solid = F.sigmoid(merge_solid)

#         fea_reduce1 = F.interpolate(fea_reduce1, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_reduce2 = F.interpolate(fea_reduce2, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_reduce3 = F.interpolate(fea_reduce3, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_reduce4 = F.interpolate(fea_reduce4, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_add1 = torch.add(merge_solid5, fea_reduce1)
#         fea_add2 = torch.add(merge_solid5, fea_reduce2)
#         fea_add3 = torch.add(merge_solid5, fea_reduce3)
#         fea_add4 = torch.add(merge_solid5, fea_reduce4)
#         fea_add1 = self.conv_add1(fea_add1)
#         fea_add1 = self.relu(fea_add1)
#         fea_add2 = self.conv_add2(fea_add2)
#         fea_add2 = self.relu(fea_add2)
#         fea_add3 = self.conv_add3(fea_add3)
#         fea_add3 = self.relu(fea_add3)
#         fea_add4 = self.conv_add4(fea_add4)
#         fea_add4 = self.relu(fea_add4)
#         feasum_out = torch.cat((fea_add1, fea_add2, fea_add3, fea_add4), 1)
#         feasum_out = self.conv_sum_out(feasum_out)
#         feasum_out = self.score_sum_out(feasum_out, x_size)
#         feasum_out = F.sigmoid(feasum_out)

#         return feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4


# class KRN_edge(nn.Module):
#     def __init__(self, base_model_cfg, base, convert_layers, score_layers):
#         super(KRN_edge, self).__init__()
#         self.base_model_cfg = base_model_cfg
#         self.base = base  # 基本网络是一样的

#         #self.deep_pool = nn.ModuleList(deep_pool_layers)
#         self.score = score_layers
#         if self.base_model_cfg == 'resnet':
#             self.convert = convert_layers

#         # 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]]
#         self.DeepPool_solid1 = DeepPoolLayer_first(512, 512, False, True)
#         self.DeepPool_solid2 = DeepPoolLayer_first(512, 256, True, True)
#         self.DeepPool_solid3 = DeepPoolLayer_first(256, 256, True, True)
#         self.DeepPool_solid4 = DeepPoolLayer_first(256, 128, True, True)
#         self.DeepPool_solid5 = DeepPoolLayer_first(128, 128, False, False)

#         self.DeepPool_contour1 = DeepPoolLayer_first(512, 512, False, True)
#         self.DeepPool_contour2 = DeepPoolLayer_first(512, 256, True, True)
#         self.DeepPool_contour3 = DeepPoolLayer_first(256, 256, True, True)
#         self.DeepPool_contour4 = DeepPoolLayer_first(256, 128, True, True)
#         self.DeepPool_contour5 = DeepPoolLayer_first(128, 128, False, False)

#         self.relu = nn.ReLU()
#         self.conv_reduce1 = nn.Conv2d(512, 128, 1, 1, 1, bias=False)
#         self.conv_reduce2 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)
#         self.conv_reduce3 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)

#         self.score_solid = ScoreLayer(128)
#         self.score_contour = ScoreLayer(128)

#         self.score_solid1 = ScoreLayer(512)
#         self.score_contour1 = ScoreLayer(512)

#         self.score_solid2 = ScoreLayer(256)
#         self.score_contour2 = ScoreLayer(256)

#         self.score_solid3 = ScoreLayer(256)
#         self.score_contour3 = ScoreLayer(256)

#         self.score_solid4 = ScoreLayer(128)
#         self.score_contour4 = ScoreLayer(128)

#         self.score_solid = ScoreLayer(128)
#         self.score_contour = ScoreLayer(128)
#         self.score_sum_out = ScoreLayer(128)

#         self.conv_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
#         self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
#         self.conv_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
#         self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

#         self.conv_add1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_add2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_add3 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_add4 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
#         self.conv_sum_out = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

#     def forward(self, x):
#         x_size = x.size()
#         conv2merge, infos = self.base(x)  #
#         if self.base_model_cfg == 'resnet':
#             conv2merge = self.convert(conv2merge)
#         conv2merge = conv2merge[::-1]

#         merge_contour1 = self.conv_1(conv2merge[1])
#         merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
#         out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
#         out_merge_contour1 = self.score_contour1(merge_contour1, x_size)
#         out_merge_solid1 = F.sigmoid(out_merge_solid1)
#         out_merge_contour1 = F.sigmoid(out_merge_contour1)
#         # merge_contour1, merge_solid1 = self.fuse1(merge_contour1, merge_solid1)
#         fea_reduce1 = self.conv_reduce1(merge_solid1)
#         fea_reduce1 = self.relu(fea_reduce1)

#         merge_contour2 = self.conv_2(conv2merge[2])
#         merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
#         out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
#         out_merge_contour2 = self.score_contour2(merge_contour2, x_size)
#         out_merge_solid2 = F.sigmoid(out_merge_solid2)
#         out_merge_contour2 = F.sigmoid(out_merge_contour2)
#         # merge_contour2, merge_solid2 = self.fuse2(merge_contour2, merge_solid2)
#         fea_reduce2 = self.conv_reduce2(merge_solid2)
#         fea_reduce2 = self.relu(fea_reduce2)

#         merge_contour3 = self.conv_3(conv2merge[3])
#         merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
#         out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
#         out_merge_contour3 = self.score_contour3(merge_contour3, x_size)
#         out_merge_solid3 = F.sigmoid(out_merge_solid3)
#         out_merge_contour3 = F.sigmoid(out_merge_contour3)
#         # merge_contour3, merge_solid3 = self.fuse3(merge_contour3, merge_solid3)
#         fea_reduce3 = self.conv_reduce3(merge_solid3)
#         fea_reduce3 = self.relu(fea_reduce3)

#         merge_contour4 = self.conv_4(conv2merge[4])
#         merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
#         out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
#         out_merge_contour4 = self.score_contour4(merge_contour4, x_size)
#         out_merge_solid4 = F.sigmoid(out_merge_solid4)
#         out_merge_contour4 = F.sigmoid(out_merge_contour4)
#         # merge_contour4, merge_solid4 = self.fuse4(merge_contour4, merge_solid4)
#         fea_reduce4 = merge_solid4

#         merge_solid5 = self.DeepPool_solid5(merge_solid4)
#         merge_solid = self.score_solid(merge_solid5, x_size)
#         merge_solid = F.sigmoid(merge_solid)

#         fea_reduce1 = F.interpolate(fea_reduce1, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_reduce2 = F.interpolate(fea_reduce2, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_reduce3 = F.interpolate(fea_reduce3, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_reduce4 = F.interpolate(fea_reduce4, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
#         fea_add1 = torch.add(merge_solid5, fea_reduce1)
#         fea_add2 = torch.add(merge_solid5, fea_reduce2)
#         fea_add3 = torch.add(merge_solid5, fea_reduce3)
#         fea_add4 = torch.add(merge_solid5, fea_reduce4)
#         fea_add1 = self.conv_add1(fea_add1)
#         fea_add1 = self.relu(fea_add1)
#         fea_add2 = self.conv_add2(fea_add2)
#         fea_add2 = self.relu(fea_add2)
#         fea_add3 = self.conv_add3(fea_add3)
#         fea_add3 = self.relu(fea_add3)
#         fea_add4 = self.conv_add4(fea_add4)
#         fea_add4 = self.relu(fea_add4)
#         feasum_out = torch.cat((fea_add1, fea_add2, fea_add3, fea_add4), 1)
#         feasum_out = self.conv_sum_out(feasum_out)
#         feasum_out = self.score_sum_out(feasum_out, x_size)
#         feasum_out = F.sigmoid(feasum_out)

#         return feasum_out, merge_solid, out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4

# def build_model(base_model_cfg='vgg'):
#     if base_model_cfg == 'resnet':
#         return KRN(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))


# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(0, 0.01)
#         if m.bias is not None:
#             m.bias.data.zero_()
            
# class KRN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = KRN('resnet', *extra_layer('resnet', resnet50_locate()))
#         self.net = self.net.cuda()
#         self.net.eval()
#         self.net.load_state_dict(torch.load(self.config.clm_model))

#         self.net_hou = KRN_edge('resnet', *extra_layer('resnet', resnet50_locate()))
#         self.net_hou = self.net_hou.cuda()
#         self.net_hou.eval()
#         self.net_hou.load_state_dict(torch.load(self.config.fsm_model))
        