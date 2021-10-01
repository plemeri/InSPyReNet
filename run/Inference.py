import torch
import yaml
import os
import argparse
import tqdm
import sys
import cv2

import torch.nn.functional as F
import numpy as np

from PIL import Image
from easydict import EasyDict as ed

import torch.utils.data as data
import torchvision.transforms as transforms

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.custom_transforms import *
from lib import *
from utils.dataloader import *
from utils.utils import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/RIUNet.yaml')
    parser.add_argument('--source', type=str)
    parser.add_argument('--type', type=str, choices=['rgba', 'map'], default='map')
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def test(opt, args):
    model = eval(opt.Model.name)(channels=opt.Model.channels, pretrained=opt.Model.pretrained)
    model.load_state_dict(torch.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)
    model.cuda()
    model.eval()    

    transform = eval(opt.Test.Dataset.type).get_transform(opt.Test.Dataset.transform_list)
    
    if os.path.isdir(args.source):
        source_dir = args.source
        source_list = os.listdir(args.source)

        save_dir = os.path.join('results', args.source.split(os.sep)[-1])

    elif os.path.isfile(args.source):
        source_dir = os.path.split(args.source)[0]
        source_list = [os.path.split(args.source)[1]]

        save_dir = 'results'
    
    else:
        return
    
    os.makedirs(save_dir, exist_ok=True)

    if args.verbose is True:
        samples = tqdm.tqdm(enumerate(source_list), desc='Inference', total=len(source_list), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        samples = enumerate(source_list)
        
    for i, sample in samples:
        img = Image.open(os.path.join(source_dir, sample))

        timg = transform({'image': img})['image'].unsqueeze(0)
        
        out = model(timg.cuda())['pred']
        out = F.interpolate(out, img.size[::-1], mode='bilinear', align_corners=True)

        out = out.data.cpu()
        out = torch.sigmoid(out)
        out = out.numpy().squeeze()
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        out = (out * 255).astype(np.uint8)
        
        if args.type == 'map':
            img = out
        elif args.type == 'rgba':
            img = np.array(img)
            r, g, b = cv2.split(img)
            img = cv2.merge([r, g, b, out])

        Image.fromarray(img).save(os.path.join(save_dir, os.path.splitext(sample)[0] + '.png'))

if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    test(opt, args)
