import os
import argparse
import yaml

from easydict import EasyDict as ed

from run import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/UACANet-L.yaml')
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))

    train(opt)
    test(opt)
    evaluate(opt)
