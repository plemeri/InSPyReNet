import argparse

from utils.misc import *
from run import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    
    train(opt, args)
    if args.local_rank <= 0:
        test(opt, args)
        evaluate(opt, args)
