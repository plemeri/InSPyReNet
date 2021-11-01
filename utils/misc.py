import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from easydict import EasyDict as ed

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

def debug_tile(deblist, size=(100, 100)):
    debugs = []
    for debs in deblist:
        debug = []
        for deb in debs:
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = ((log - log.min()) / (log.max() - log.min()) * 255).astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)

def unfold(x, patch_size=256):
    b, c, h, w = x.shape
    stride = patch_size // 2
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=stride)

    patches = unfold(x)
    patches = patches.reshape(
        c, patch_size, patch_size, -1).contiguous().permute(3, 0, 1, 2)

    return patches, (b, c, h, w)


def fold(patches, target_shape, patch_size=256):
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

if __name__ == "__main__":
    x = torch.rand(4, 3, 576, 576)