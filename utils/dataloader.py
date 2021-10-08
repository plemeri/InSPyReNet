import os
import cv2

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

from utils.custom_transforms import *

class RGB_Dataset(data.Dataset):
    def __init__(self, root, transform_list):
        image_root, gt_root = os.path.join(root, 'images'), os.path.join(root, 'masks')

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.gts = sorted(self.gts)
        
        self.filter_files()
        
        self.size = len(self.images)
        self.transform = self.get_transform(transform_list)

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
            
        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape}

        sample = self.transform(sample)
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size

class RGBD_Dataset(data.Dataset):
    def __init__(self, root, transform_list):
        image_root = os.path.join(root, 'RGB')
        gt_root = os.path.join(root, 'GT')
        depth_root = os.path.join(root, 'depth')

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = sorted(self.depths)
        
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.gts = sorted(self.gts)
        
        self.filter_files()
        
        self.size = len(self.images)
        self.transform = self.get_transform(transform_list)

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        depth = Image.open(self.depths[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split(os.sep)[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
                
        sample = {'image': image, 'gt': gt, 'depth': depth, 'name': name, 'shape': shape}
        sample = self.transform(sample)
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts) == len(self.depths)
        images, gts, depths = [], [], []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img, gt, depth = Image.open(img_path), Image.open(gt_path), Image.open(depth_path)
            if img.size == gt.size  == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images, self.gts, self.depths = images, gts, depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size