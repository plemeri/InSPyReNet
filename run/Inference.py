import torch
import os
import argparse
import tqdm
import sys
import cv2

import torch.nn.functional as F
import numpy as np

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.utils import *
from utils.dataloader import *
from utils.custom_transforms import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/RIUNet.yaml')
    parser.add_argument('--source', type=str)
    parser.add_argument('--type', type=str,
                        choices=['rgba', 'map'], default='map')
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


def inference(opt, args):
    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                 pretrained=opt.Model.pretrained)
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)
    model.cuda()
    model.eval()

    transform = eval(opt.Test.Dataset.type).get_transform(
        opt.Test.Dataset.transform_list)

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
        sources = tqdm.tqdm(enumerate(source_list), desc='Inference', total=len(
            source_list), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        sources = enumerate(source_list)

    for i, source in sources:
        img = Image.open(os.path.join(source_dir, source)).convert('RGB')
        sample = {'image': img}
        sample = transform(sample)
        sample['image'] = sample['image'].unsqueeze(0)
        sample = to_cuda(sample)

        out = model(sample)
        out['pred'] = F.interpolate(
            out['pred'], img.size[::-1], mode='bilinear', align_corners=True)
        out['pred'] = out['pred'].data.cpu()
        out['pred'] = torch.sigmoid(out['pred'])
        out['pred'] = out['pred'].numpy().squeeze()
        out['pred'] = (out['pred'] - out['pred'].min()) / \
            (out['pred'].max() - out['pred'].min() + 1e-8)
        out['pred'] = (out['pred'] * 255).astype(np.uint8)

        if args.type == 'map':
            img = out['pred']
        elif args.type == 'rgba':
            img = np.array(img)
            r, g, b = cv2.split(img)
            img = cv2.merge([r, g, b, out['pred']])

        Image.fromarray(img).save(os.path.join(
            save_dir, os.path.splitext(source)[0] + '.png'))


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    inference(opt, args)
