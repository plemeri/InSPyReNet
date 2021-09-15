import os
import argparse
import yaml

from easydict import EasyDict as ed

from run import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/UACANet-L.yaml')
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))

    train(opt, verbose=args.verbose)
    test(opt, verbose=args.verbose)
    evaluate(opt, verbose=args.verbose)
