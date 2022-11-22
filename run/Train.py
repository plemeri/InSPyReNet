import os
import cv2
import sys
import tqdm
import torch
import datetime

import torch.nn as nn
import torch.distributed as dist
import torch.cuda as cuda

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from lib.optim import *
from data.dataloader import *
from utils.misc import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def train(opt, args):
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, 
        sets=opt.Train.Dataset.sets,
        tfs=opt.Train.Dataset.transforms)

    if args.device_num > 1:
        cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num, timeout=datetime.timedelta(seconds=3600))
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=opt.Train.Dataloader.batch_size,
                            shuffle=train_sampler is None,
                            sampler=train_sampler,
                            num_workers=opt.Train.Dataloader.num_workers,
                            pin_memory=opt.Train.Dataloader.pin_memory,
                            drop_last=True)

    model_ckpt = None
    state_ckpt = None
    
    if args.resume is True:
        if os.path.isfile(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth')):
            model_ckpt = torch.load(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'), map_location='cpu')
            if args.local_rank <= 0:
                print('Resume from checkpoint')
        if os.path.isfile(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'state.pth')):
            state_ckpt = torch.load(os.path.join(opt.Train.Checkpoint.checkpoint_dir,  'state.pth'), map_location='cpu')
            if args.local_rank <= 0:
                print('Resume from state')
        
    model = eval(opt.Model.name)(**opt.Model)
    if model_ckpt is not None:
        model.load_state_dict(model_ckpt)

    if args.device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = model.cuda()

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params}, {
        'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    
    optimizer = eval(opt.Train.Optimizer.type)(
        params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    
    if state_ckpt is not None:
        optimizer.load_state_dict(state_ckpt['optimizer'])
    
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                                minimum_lr=opt.Train.Scheduler.minimum_lr,
                                                max_iteration=len(train_loader) * opt.Train.Scheduler.epoch,
                                                warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    if state_ckpt is not None:
        scheduler.load_state_dict(state_ckpt['scheduler'])

    model.train()

    start = 1
    if state_ckpt is not None:
        start = state_ckpt['epoch']
        
    epoch_iter = range(start, opt.Train.Scheduler.epoch + 1)
    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(epoch_iter, desc='Epoch', total=opt.Train.Scheduler.epoch, initial=start - 1,
                                position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')

    for epoch in epoch_iter:
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
            if args.device_num > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True and scaler is not None:
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
                if args.device_num > 1:
                    model_ckpt = model.module.state_dict()  
                else:
                    model_ckpt = model.state_dict()
                    
                state_ckpt = {'epoch': epoch + 1,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}
                
                torch.save(model_ckpt, os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))
                torch.save(state_ckpt, os.path.join(opt.Train.Checkpoint.checkpoint_dir,  'state.pth'))
                
            if args.debug is True:
                debout = debug_tile(sum([out[k] for k in opt.Train.Debug.keys], []), activation=torch.sigmoid)
                cv2.imwrite(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)
    
    if args.local_rank <= 0:
        torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                    os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)
    train(opt, args)
