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

class resize:
    def __init__(self, size=384, pad=False):
        if hasattr(size, '__iter__'):
            self.size = size[::-1]
        else:
            self.size = size
            
        self.pad = pad
            
    def __call__(self, sample):
        if hasattr(self.size, '__iter__'):
            size = self.size
        else:
            size = list(sample['image'].size)
            if (size[0] >= size[1]) and size[1] > self.size: 
                size[0] = size[0] / (size[1] / self.size)
                size[1] = self.size
            elif (size[1] > size[0]) and size[0] > self.size:
                size[1] = size[1] / (size[0] / self.size)
                size[0] = self.size
            size = (int(round(size[0] / 32)) * 32, int(round(size[1] / 32)) * 32)
        
        if 'image' in sample.keys():
            sample['image'] = sample['image'].resize(size, Image.BILINEAR)
        if 'gt' in sample.keys():
            sample['gt'] = sample['gt'].resize(size, Image.NEAREST)
        if 'depth' in sample.keys():
            sample['depth'] = sample['depth'].resize(size, Image.NEAREST)
            
        # if self.pad is True:
        #     for key in ['image', 'gt', 'depth']:
        #         if key in sample.keys():
        #             sample[key] = Image.new(sample[key].mode, (size[0] + 32, size[1] + 32)).paste(sample[key], (16, 16))
            
        return sample
    
class cvtcolor:
    def __init__(self):
        pass
    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = Image.fromarray(np.array(sample['image'])[:, :, ::-1])

        return sample
    
class dynamic_resize:
    def __init__(self, patch_size, stride):
        assert patch_size % stride == 0
        self.patch_size = patch_size
        self.stride = patch_size // stride

    def __call__(self, sample):
        if 'image' in sample.keys():
            ar = sample['image'].size[0] / sample['image'].size[1]
            hx, wx = 1, 1
            if ar > 1:
                hx = round(self.stride * ar) / self.stride
            else:
                wx = round(self.stride / ar) / self.stride
            
            size = (int(self.patch_size * hx), int(self.patch_size * wx))
            
            sample['image'] = sample['image'].resize(size, Image.BILINEAR)
            sample['patch_size'] = self.patch_size
            sample['stride'] = self.patch_size // self.stride

        return sample

class random_dilation_erosion:
    def __init__(self, kernel_range):
        self.kernel_range = kernel_range

    def __call__(self, sample):
        gt = sample['gt']
        gt = np.array(gt)
        iou_max = 1.0
        iou_min = 0.8
        iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
        gt = modify_boundary(gt, iou_target=iou_target)
        sample['depth'] = Image.fromarray(gt)

        return sample

class random_scale_crop:
    def __init__(self, range=[0.75, 1.25]):
        self.range = range

    def __call__(self, sample):
        scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt', 'depth']:
                    base_size = sample[key].size

                    scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                    sample[key] = sample[key].resize(scale_size)

                    lf = (sample[key].size[0] - base_size[0]) // 2
                    up = (sample[key].size[1] - base_size[1]) // 2
                    rg = (sample[key].size[0] + base_size[0]) // 2
                    lw = (sample[key].size[1] + base_size[1]) // 2

                    border = -min(0, min(lf, up))
                    sample[key] = ImageOps.expand(sample[key], border=border, fill=255 if key == 'depth' else None)
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
            if key in ['image', 'gt', 'depth']:
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
                if key in ['image', 'gt', 'depth']:
                    base_size = sample[key].size
                    sample[key] = sample[key].rotate(rot, expand=True, fillcolor=255 if key == 'depth' else None)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

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

class random_gaussian_blur:
    def __init__(self):
        pass

    def __call__(self, sample):
        if np.random.random() < 0.5 and 'image' in sample.keys():
            sample['image'] = sample['image'].filter(ImageFilter.GaussianBlur(radius=np.random.random()))

        return sample

class random_lens_generation:
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, sample):
        sample['lens'] = np.ones_like(np.array(sample['image'])).astype(np.uint8) * 255
        if np.random.random() < 0.5:
            sample['lens']

        return sample


class histogram_equalization:
    def __init__(self):
        pass
        
    def __call__(self, sample):
        if 'depth' in sample.keys():
            sample['depth'] = Image.fromarray(cv2.equalizeHist(np.array(sample['depth'])))

        return sample
    
class clahe:
    def __init__(self, clip=4.0, grid=(8, 8)):
        self.eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        
    def __call__(self, sample):
        if 'depth' in sample.keys():
            sample['depth'] = Image.fromarray(self.eq.apply(np.array(sample['depth'])))

        return sample
    
class random_gamma_corruption:
    def __init__(self):
        pass
        
    def __call__(self, sample):
        if 'depth' in sample.keys():
            if np.random.random() > .5:
                depth = np.array(sample['depth'])
                depth = (depth / 255) ** (np.random.random() * .4 + .8)
                depth = (depth * 255).astype(np.uint8)
                sample['depth'] = depth

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image', 'gt', 'depth']:
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

        if 'gt' in sample.keys():
            sample['gt'] /= self.div

        if 'depth' in sample.keys():
            sample['depth'] /= self.div

        return sample
class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].transpose((2, 0, 1))
            sample['image'] = torch.from_numpy(sample['image']).float()
            
        if 'gt' in sample.keys():
            sample['gt'] = torch.from_numpy(sample['gt'])
            sample['gt'] = sample['gt'].unsqueeze(dim=0)

        if 'depth' in sample.keys():
            # sample['depth'] = sample['depth'].transpose((2, 0, 1))
            # sample['depth'] = torch.from_numpy(sample['depth']).float()
            sample['depth'] = torch.from_numpy(sample['depth'])
            sample['depth'] = sample['depth'].unsqueeze(dim=0)

        return sample
