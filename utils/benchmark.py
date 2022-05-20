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
    parser.add_argument('--input_size', type=int, nargs='+', default=[384, 384])
    parser.add_argument('--verbose',    action='store_true', default=False)
    return parser.parse_args()

def benchmark(opt, args):
    model = Simplify(eval(opt.Model.name)(**opt.Model))
    model = model.cuda()
    
    input = torch.rand(1, 3, *args.input_size)   
    input = input.cuda()
    
    macs, params = profile(model, inputs=(input, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for i in range(10):
            out = model(input)
            end.record()
        
    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Model:', opt.Model.name)
    print('MACs:', macs, 'Params:', params)
    print('Throughput:', start.elapsed_time(end) / 10, 'msec')

if __name__ == '__main__':
    args = _args()
    opt = load_config(args.config)
    benchmark(opt, args)
