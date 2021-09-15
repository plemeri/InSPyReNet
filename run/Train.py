from enum import auto
import os
import torch
import argparse
import yaml
import tqdm
import sys

import cv2
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.cuda.amp import GradScaler, autocast
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from lib.optim import *
from utils.dataloader import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/SwinYNet.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def train(opt, local_rank=-1, verbose=False):
    # device_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    device_ids = [0]
    device_num = len(device_ids)

    train_dataset = eval(opt.Train.Dataset.type)(image_root=os.path.join(opt.Train.Dataset.root, 'images'),
                                                 gt_root=os.path.join(opt.Train.Dataset.root, 'masks'),
                                                 transform_list=opt.Train.Dataset.transform_list)

    if device_num > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=opt.Train.Dataloader.batch_size,
                                   shuffle=train_sampler is None,
                                   sampler=train_sampler,
                                   num_workers=opt.Train.Dataloader.num_workers,
                                   pin_memory=opt.Train.Dataloader.pin_memory,
                                   drop_last=True)

    model = eval(opt.Model.name)(channels=opt.Model.channels, pretrained=opt.Model.pretrained)

    if device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = model.cuda()

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            if 'backbone.layer' in name:
                backbone_params.append(param)
            else:
                pass
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params}, {'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    optimizer = eval(opt.Train.Optimizer.type)(params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                                          minimum_lr=opt.Train.Scheduler.minimum_lr,
                                                          max_iteration=len(train_loader) * opt.Train.Scheduler.epoch,
                                                          warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    model.train()

    if local_rank <= 0 and verbose is True:
        epoch_iter = tqdm.tqdm(range(1, opt.Train.Scheduler.epoch + 1), desc='Epoch', total=opt.Train.Scheduler.epoch, position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)

    for epoch in epoch_iter:
        if local_rank <= 0 and verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
            if device_num > 1:
                train_sampler.set_epoch(epoch)
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True:
                with autocast():
                    out = model(sample['image'].cuda(), sample['gt'].cuda())

                    scaler.scale(out['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                out = model(sample['image'].cuda(), sample['gt'].cuda())
                out['loss'].backward()
                optimizer.step()
                scheduler.step()

            if local_rank <= 0 and verbose is True:
                step_iter.set_postfix({'loss': out['loss'].item()})

        if local_rank <= 0:
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                torch.save(model.module.state_dict() if device_num > 1 else model.state_dict(), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))

            ## debug
            debugs = []
            for i in range(sample['image'].shape[0]):
                debug = []

                img = sample['image'][i].permute(1, 2, 0).cpu().detach().numpy()
                img *= np.array(opt.Train.Dataset.transform_list.normalize.std)
                img += np.array(opt.Train.Dataset.transform_list.normalize.mean)
                img *= 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                debug.append(img)

                log = torch.sigmoid(out['pred'][i]).cpu().detach().numpy().squeeze()
                log *= 255
                log = log.astype(np.uint8)
                log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
                debug.append(log)

                for deb in out['debug']:
                    log2 = torch.sigmoid(deb[i]).cpu().detach().numpy().squeeze()
                    log2 = (log2 - log2.min()) / (log2.max() - log2.min())
                    log2 *= 255
                    log2 = log2.astype(np.uint8)
                    log2 = cv2.cvtColor(log2, cv2.COLOR_GRAY2RGB)
                    log2 = cv2.resize(log2, log.shape[:2])
                    debug.append(log2)

                debugs.append(np.hstack(debug))
            debout = np.vstack(debugs)
            cv2.imwrite(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)

    if local_rank <= 0:
        torch.save(model.module.state_dict() if device_num > 1 else model.state_dict(), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    train(opt, args.local_rank, args.verbose)
