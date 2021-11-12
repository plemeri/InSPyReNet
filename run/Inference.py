import torch
import os
import argparse
import tqdm
import sys
import cv2

import numpy as np

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *
from data.custom_transforms import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  type=str,            default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--source',  type=str,            default='test')
    parser.add_argument('--dest',    type=str,            default=None)
    parser.add_argument('--type',    type=str,            default='map')
    parser.add_argument('--gpu',     action='store_true', default=False)
    parser.add_argument('--jit',     action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--PM',      action='store_true', default=False)
    parser.add_argument('--MS',      action='store_true', default=False)
    return parser.parse_args()

def get_format(source):
    img_count = len([i for i in source if i.lower().endswith(('.jpg', '.png', '.jpeg'))])
    vid_count = len([i for i in source if i.lower().endswith(('.mp4', '.avi', '.mov' ))])
    
    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''

def inference(opt, args):
    model = eval(opt.Model.name)(depth=opt.Model.depth, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest.pth'), map_location=torch.device('cpu')), strict=True)
    
    if args.gpu is True:
        model.cuda()
    model.eval()
    
    if args.PM is True:
        model = InSPyReNet_PM(model, opt.Model.PM.patch_size, opt.Model.PM.stride)
        
    
    if args.MS is True:
        model = InSPyReNet_MS(model)    
    
    if args.jit is True:
        if os.path.isfile(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt')) is False:
            model = torch.jit.trace(model, torch.rand(1, 3, 384, 384).cuda())
            torch.jit.save(model, os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt'))
        
        else:
            del model
            model = torch.jit.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt'))
            if args.gpu is True:
                model.cuda()
    
    save_dir = None
    _format = None
    
    if args.source.isnumeric() is True:
        _format = 'Webcam'

    elif os.path.isdir(args.source):
        save_dir = os.path.join('results', args.source.split(os.sep)[-1])
        _format = get_format(os.listdir(args.source))

    elif os.path.isfile(args.source):
        save_dir = 'results'
        _format = get_format([args.source])
        
    if args.dest is not None:
        save_dir = args.dest
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    sample_list = eval(_format + 'Loader')(args.source, opt.Test.Dataset.transforms_PM if args.PM else opt.Test.Dataset.transforms)

    if args.verbose is True:
        samples = tqdm.tqdm(sample_list, desc='Inference', total=len(
            sample_list), position=0, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        samples = sample_list
        
    writer = None
    background = None

    for sample in samples:
        if _format == 'Video' and writer is None:
            writer = cv2.VideoWriter(os.path.join(save_dir, sample['name'] + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), sample_list.fps, sample['shape'][::-1])
            samples.total += int(sample_list.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if _format == 'Video' and sample['image'] is None:
            if writer is not None:
                writer.release()
            writer = None
            continue
        
        if args.gpu is True:
            sample = to_cuda(sample)

        with torch.no_grad():
            out = model(sample)#['image'])
        pred = to_numpy(out['pred'], sample['shape'])

        img = np.array(sample['original'])
        if args.type == 'map':
            img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
        elif args.type == 'rgba':
            r, g, b = cv2.split(img)
            pred = (pred * 255).astype(np.uint8)
            img = cv2.merge([r, g, b, pred])
        elif args.type == 'green':
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155]
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])
        elif args.type == 'blur':
            img = img * pred[..., np.newaxis] + cv2.GaussianBlur(img, (0, 0), 15) * (1 - pred[..., np.newaxis])
        elif args.type.lower().endswith(('.jpg', '.jpeg', '.png')):
            if background is None:
                background = cv2.cvtColor(cv2.imread(args.type), cv2.COLOR_BGR2RGB)
                background = cv2.resize(background, img.shape[:2][::-1])
            img = img * pred[..., np.newaxis] + background * (1 - pred[..., np.newaxis])
        img = img.astype(np.uint8)
        
        if _format == 'Image':
            Image.fromarray(img).save(os.path.join(save_dir, sample['name'] + '.png'))
            # out = model(sample)
            # for i, d in enumerate(out['gaussian']):
            #     d = to_numpy(torch.sigmoid(d), sample['shape'])
            #     img = (np.stack([d] * 3, axis=-1) * 255).astype(np.uint8)
            #     Image.fromarray(img).save(os.path.join(save_dir, str(i) + sample['name'] + '.png'))
        elif _format == 'Video' and writer is not None:
            writer.write(img)
        elif _format == 'Webcam':
            cv2.imshow('InSPyReNet', img)

if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    inference(opt, args)
