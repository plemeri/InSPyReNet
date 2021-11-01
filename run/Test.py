import torch
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser.parse_args()


def test(opt, args):
    model = eval(opt.Model.name)(depth=opt.Model.depth,
                                pretrained=False)
    model.load_state_dict(torch.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)
    model.cuda()
    model.eval()

    if args.verbose is True:
        sets = tqdm.tqdm(opt.Test.Dataset.sets, desc='Total TestSet', total=len(
            opt.Test.Dataset.sets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        sets = opt.Test.Dataset.sets

    for set in sets:
        save_path = os.path.join(opt.Test.Checkpoint.checkpoint_dir, set)

        os.makedirs(save_path, exist_ok=True)
        test_loader = ImageLoader(os.path.join(opt.Test.Dataset.root, set, 'images'), opt.Test.Dataset.transform_list)

        if args.verbose is True:
            samples = tqdm.tqdm(test_loader, desc=set + ' - Test', total=len(test_loader),
                                position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = test_loader

        for sample in samples:
            sample = to_cuda(sample)
            with torch.no_grad():
                out = model(sample['image'])
            pred = to_numpy(out, sample['shape'])
            Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(save_path, os.path.splitext(sample['name'])[0] + '.png'))


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    test(opt, args)
