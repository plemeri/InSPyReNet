from email.mime import base
import numpy as np
from PIL import Image
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from typing import Optional

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.misc import *
        
class static_resize:
    # Resize for training
    def __init__(self, size=[384, 384], base_size=None):
        self.size = size
        self.base_size = base_size
            
    def __call__(self, sample):
        sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        if 'gt' in sample.keys():
            sample['gt'] = sample['gt'].resize(self.size, Image.NEAREST)
            
        if self.base_size is not None:
            sample['image_resized'] = sample['image'].resize(self.size, Image.BILINEAR)
            if 'gt' in sample.keys():
                sample['gt_resized'] = sample['gt'].resize(self.size, Image.NEAREST)
                
        return sample

class dynamic_resize:
    def __init__(self, L=1280, base_size=[384, 384]): 
        self.L = L
        self.base_size = base_size
                    
    def __call__(self, sample):
        size = list(sample['image'].size)
        if (size[0] >= size[1]) and size[1] > self.L: 
            size[0] = size[0] / (size[1] / self.L)
            size[1] = self.L
        elif (size[1] > size[0]) and size[0] > self.L:
            size[1] = size[1] / (size[0] / self.L)
            size[0] = self.L
        size = (int(round(size[0] / 32)) * 32, int(round(size[1] / 32)) * 32)
    
        if 'image' in sample.keys():
            sample['image_resized'] = sample['image'].resize(self.base_size, Image.BILINEAR)
            sample['image'] = sample['image'].resize(size, Image.BILINEAR)
            
        if 'gt' in sample.keys():
            sample['gt_resized'] = sample['gt'].resize(self.base_size, Image.NEAREST)
            sample['gt'] = sample['gt'].resize(size, Image.NEAREST)
        
        return sample
        
class random_scale_crop:
    def __init__(self, range=[0.75, 1.25]):
        self.range = range

    def __call__(self, sample):
        scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt']:
                    base_size = sample[key].size

                    scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                    sample[key] = sample[key].resize(scale_size)

                    lf = (sample[key].size[0] - base_size[0]) // 2
                    up = (sample[key].size[1] - base_size[1]) // 2
                    rg = (sample[key].size[0] + base_size[0]) // 2
                    lw = (sample[key].size[1] + base_size[1]) // 2

                    border = -min(0, min(lf, up))
                    sample[key] = ImageOps.expand(sample[key], border=border)
                    sample[key] = sample[key].crop((lf + border, up + border, rg + border, lw + border))
        return sample

class random_flip:
    def __init__(self, lr=True, ud=True):
        self.lr = lr
        self.ud = ud

    def __call__(self, sample):
        lr = np.random.random() < 0.5 and self.lr is True
        ud = np.random.random() < 0.5 and self.ud is True

        for key in sample.keys():
            if key in ['image', 'gt']:
                sample[key] = np.array(sample[key])
                if lr:
                    sample[key] = np.fliplr(sample[key])
                if ud:
                    sample[key] = np.flipud(sample[key])
                sample[key] = Image.fromarray(sample[key])

        return sample

class random_rotate:
    def __init__(self, range=[0, 360], interval=1):
        self.range = range
        self.interval = interval

    def __call__(self, sample):
        rot = (np.random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt']:
                    base_size = sample[key].size
                    sample[key] = sample[key].rotate(rot, expand=True, fillcolor=255 if key == 'depth' else None)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample
    
class hard_region_sampling:
    def __init__(self, size=[384, 384]):
        self.size = size
        self.static_resize = static_resize(size)

    def __call__(self, sample):
        image = np.array(sample['image'])
        mask  = np.array(sample['gt'])
        
        if np.random.random() < .5:
            x           = torch.tensor(mask.squeeze()).unsqueeze(0).unsqueeze(0).float() / 255
            hard_region = torch.abs(F.avg_pool2d(x, kernel_size=31, stride=1, padding=15) - x)
            target      = torch.where(hard_region > .5)[2:]
            
            if len(target[0]) != 0:
                idx         = torch.randint(0, len(target[0]), (1,))[0]
                coord       = [target[0][idx], target[1][idx]]
                
                if   coord[0] + (self.size[0] // 2) > mask.shape[0]:
                    coord[0] = mask.shape[0] - (self.size[0] // 2)
                elif coord[0] - (self.size[0] // 2) < 0:
                    coord[0] = (self.size[0] // 2)

                if   coord[1] + (self.size[1] // 2) > mask.shape[1]:
                    coord[1] = mask.shape[1] - (self.size[1] // 2)
                elif coord[1] - (self.size[1] // 2) < 0:
                    coord[1] = (self.size[1] // 2)

                top_left     = (coord[0] - (self.size[0] // 2), coord[1] - (self.size[1] // 2))
                bottom_right = (coord[0] + (self.size[0] // 2), coord[1] + (self.size[1] // 2))
                
                sample['image'] = Image.fromarray(image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
                sample['gt']    = Image.fromarray( mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
            else:
                sample = self.static_resize(sample)    
        else:
            sample = self.static_resize(sample)

        return sample

class random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, sample):
        if 'image' in sample.keys():
            np.random.shuffle(self.enhance_method)

            for method in self.enhance_method:
                if np.random.random() > 0.5:
                    enhancer = method(sample['image'])
                    factor = float(1 + np.random.random() / 10)
                    sample['image'] = enhancer.enhance(factor)

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image', 'image_resized', 'gt', 'gt_resized']:
                sample[key] = np.array(sample[key], dtype=np.float32)

        return sample

class normalize:
    def __init__(self, mean: Optional[list]=None, std: Optional[list]=None, div=255):
        self.mean = mean if mean is not None else 0.0
        self.std = std if std is not None else 1.0
        self.div = div
        
    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] /= self.div
            sample['image'] -= self.mean
            sample['image'] /= self.std
            
        if 'image_resized' in sample.keys():
            sample['image_resized'] /= self.div
            sample['image_resized'] -= self.mean
            sample['image_resized'] /= self.std

        if 'gt' in sample.keys():
            sample['gt'] /= self.div
            
        if 'gt_resized' in sample.keys():
            sample['gt_resized'] /= self.div

        return sample
class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].transpose((2, 0, 1))
            sample['image'] = torch.from_numpy(sample['image']).float()
        
        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].transpose((2, 0, 1))
            sample['image_resized'] = torch.from_numpy(sample['image_resized']).float()
            
        if 'gt' in sample.keys():
            sample['gt'] = torch.from_numpy(sample['gt'])
            sample['gt'] = sample['gt'].unsqueeze(dim=0)
            
        if 'gt_resized' in sample.keys():
            sample['gt_resized'] = torch.from_numpy(sample['gt_resized'])
            sample['gt_resized'] = sample['gt_resized'].unsqueeze(dim=0)

        return sample
