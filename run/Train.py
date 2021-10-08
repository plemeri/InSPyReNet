import os
import torch
import argparse
import tqdm
import sys

import cv2
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Adam, SGD
from torch.cuda.amp import GradScaler, autocast

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.dataloader import *
from lib.optim import *
from lib import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def train(opt, args):
    device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    device_num = len(device_ids)

    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, transform_list=opt.Train.Dataset.transform_list)

    if device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=opt.Train.Dataloader.batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=opt.Train.Dataloader.num_workers,
                                    pin_memory=opt.Train.Dataloader.pin_memory,
                                    drop_last=True)

    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                 pretrained=opt.Model.pretrained)

    if device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[
                                                    args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
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

    params_list = [{'params': backbone_params}, {
        'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    optimizer = eval(opt.Train.Optimizer.type)(
        params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                               minimum_lr=opt.Train.Scheduler.minimum_lr,
                                               max_iteration=len(
                                                   train_loader) * opt.Train.Scheduler.epoch,
                                               warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    model.train()

    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(range(1, opt.Train.Scheduler.epoch + 1), desc='Epoch', total=opt.Train.Scheduler.epoch,
                               position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)

    for epoch in epoch_iter:
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
            if device_num > 1:
                train_sampler.set_epoch(epoch)
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True:
                with autocast():
                    sample = to_cuda(sample)
                    out = model(sample)

                    scaler.scale(out['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                sample = to_cuda(sample)
                out = model(sample)
                out['loss'].backward()
                optimizer.step()
                scheduler.step()

            if args.local_rank <= 0 and args.verbose is True:
                step_iter.set_postfix({'loss': out['loss'].item()})

        if args.local_rank <= 0:
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(
                opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                torch.save(model.module.state_dict() if device_num > 1 else model.state_dict(
                ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))

            if args.debug is True:
                debout = debug_tile(out)
                cv2.imwrite(os.path.join(
                    opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict() if device_num > 1 else model.state_dict(
        ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = _args()
    opt = load_config(args.config)
    train(opt, args)
