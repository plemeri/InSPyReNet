import torch
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.utils.data.dataloader import DataLoader

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--PM',      action='store_true', default=False)
    return parser.parse_args()


def test(opt, args):
    model = eval(opt.Model.name)(**opt.Model)
    model.load_state_dict(torch.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)
    
    if args.PM is True:
        if 'InSPyRe' in opt.Model.name:
            model = PPM(model)
        else:
            model = SPM(model) #, opt.PM.patch_size, opt.PM.stride)
        
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
        # test_loader = ImageLoader(os.path.join(opt.Test.Dataset.root, set, 'images'), opt.Test.Dataset.transforms)
        test_dataset = eval(opt.Test.Dataset.type)(opt.Test.Dataset.root, [set], opt.Test.Dataset.transforms if args.PM is not True else opt.Test.Dataset.transforms_PM)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                num_workers=opt.Test.Dataloader.num_workers,
                                pin_memory=opt.Test.Dataloader.pin_memory)

        if args.verbose is True:
            samples = tqdm.tqdm(test_loader, desc=set + ' - Test', total=len(test_loader),
                                position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = test_loader

        for sample in samples:
            sample = to_cuda(sample)
            with torch.no_grad():
                out = model(sample)
            pred = to_numpy(out['pred'], sample['shape'])
            Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(save_path, sample['name'][0] + '.png'))


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    test(opt, args)
