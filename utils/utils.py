import os
import sys
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from easydict import EasyDict as ed
import torchvision.transforms as transforms

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.custom_transforms import *

def get_transform(transform_list):
    tfs = []
    for key, value in zip(transform_list.keys(), transform_list.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        tfs.append(tf)
    return transforms.Compose(tfs)

def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    return cfg

def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample

def to_numpy(pred, shape):
    pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
    pred = pred.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred

def patch(x, patch_size=256):
    b, c, h, w = x.shape
    stride = patch_size // 2
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=stride)

    patches = unfold(x)
    patches = patches.reshape(
        c, patch_size, patch_size, -1).contiguous().permute(3, 0, 1, 2)

    return patches, (b, c, h, w)


def stitch(patches, target_shape, patch_size=256):
    b, c, h, w = target_shape
    stride = patch_size // 2
    fold = nn.Fold(output_size=(h, w), kernel_size=(
        patch_size,) * 2, stride=stride)
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=stride)

    patches = patches.permute(1, 2, 3, 0).reshape(
        b, c * patch_size ** 2, patches.shape[0] // b)

    weight = torch.ones(*target_shape).to(patches.device)
    weight = unfold(weight)

    out = fold(patches) # / fold(weight)

    return out

def patch_max(x, patch_size=256):
    stride = patch_size // 2
    assert stride != 0
    b, c, h, w = x.shape
    assert h // stride != 0
    assert w // stride != 0
    
    ph, pw = h // stride - 1, w // stride - 1
    patches = torch.zeros(b * ph * pw, c, patch_size, patch_size).to(x.device)
    for i in range(ph):
        for j in range(pw):
            start = (j + pw * i)# * stride
            end = start + c# + stride
            patches[start:end] = x[:, :, i * stride: (i + 2) * stride, j * stride: (j + 2) * stride]
    return patches, (b, c, h, w)

def stitch_max(patches, target_shape, patch_size=256):
    b, c, h, w = target_shape
    stride = patch_size // 2
    assert stride != 0
    assert h // stride != 0
    assert w // stride != 0
    
    ph, pw = h // stride - 1, w // stride - 1
      
    out = torch.zeros(b, c, h, w).to(patches.device)
    for i in range(ph):
            for j in range(pw):
                start = (j + pw * i)# * stride
                end = start + c# + stride
                
                tgt = out[:, :, i * stride: (i + 2) * stride, j * stride: (j + 2) * stride]
                src = patches[start:end]
                
                patch_neg = -torch.max(F.relu(-tgt), F.relu(-src))
                patch_pos = torch.max(F.relu(tgt), F.relu(src))
                patch = patch_neg + patch_pos
                
                patch = torch.max(out[:, :, i * stride: (i + 2) * stride, j * stride: (j + 2) * stride].abs(), patches[start:end])
                out[:, :, i * stride: (i + 2) * stride, j * stride: (j + 2) * stride] = patch
    return out

def stitch_avg(patches, target_shape, patch_size=256):
    b, c, h, w = target_shape
    stride = patch_size // 2
    assert stride != 0
    assert h // stride != 0
    assert w // stride != 0
    
    ph, pw = h // stride - 1, w // stride - 1
    
    out = torch.zeros(b, c, h, w).to(patches.device)
    wgt = torch.zeros(b, c, h, w).to(patches.device)
    for i in range(ph):
            for j in range(pw):
                start = (j + pw * i)# * stride
                end = start + c# + stride
                out[:, :, i * stride: (i + 2) * stride, j * stride: (j + 2) * stride] += patches[start:end]
                wgt[:, :, i * stride: (i + 2) * stride, j * stride: (j + 2) * stride] += torch.ones_like(patches[start:end]).to(patches.device)
    return out# / wgt


def stitch_avg_mod(patches, target_shape, patch_size=256):
    b, c, h, w = target_shape
    stride = patch_size // 2
    assert stride != 0
    assert h // stride != 0
    assert w // stride != 0
    
    ph, pw = h // stride - 1, w // stride - 1
    
    out = torch.zeros(b, c, h, w).to(patches.device)
    for i in range(ph):
            for j in range(pw):
                cstart = (j + pw * i)# * stride
                cend = cstart + c# + stride
                
                hstart = stride // 2
                hend = patch_size - (hstart)
                
                wstart = stride // 2
                wend = patch_size - (wstart)
                
                if i == 0:
                    hstart = 0
                if i == ph - 1:
                    hend = patch_size
                
                if j == 0:
                    wstart = 0
                if j == pw - 1:
                    wend = patch_size
                    
                out[:, :, i * stride + hstart: i * stride + hend, j * stride + wstart: j * stride + wend] += patches[cstart:cend, :, hstart:hend, wstart:wend]
    return out # / wgt

def debug_tile(out, size=(100, 100)):
    debugs = []
    for debs in out['debug']:
        debug = []
        for deb in debs:
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = (log - log.min()) / (log.max() - log.min())
            log *= 255
            log = log.astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)

if __name__ == "__main__":
    x = torch.rand(4, 3, 576, 576)
    y, shape = patch_max(x, 384)
    x_ = stitch_max(y, shape, 384)
    
    print(x == x_)
    
    