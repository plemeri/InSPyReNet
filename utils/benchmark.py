import argparse
import torch
import os
import sys
from thop import profile, clever_format

import warnings
warnings.filterwarnings("ignore")

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from lib.optim import *
from data.dataloader import *
from utils.misc import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser.parse_args()

def benchmark(opt, args):
    model = eval(opt.Model.name)(depth=opt.Model.depth, pretrained=False)
    input = torch.rand(1, 3, *opt.Test.Dataset.transform_list.resize.size)
    
    macs, params = profile(model, inputs=(input, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    
    print('Model:', opt.Model.name)
    print('MACs:', macs, 'Params:', params)

if __name__ == '__main__':
    args = _args()
    opt = load_config(args.config)
    benchmark(opt, args)
