import os
import argparse
import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.misc import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default=['configs/InSPyReNet_SwinB.yaml'])
    parser.add_argument('--datasets', type=str, nargs='+', default=['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S'])
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser.parse_args()


def draw_figure(opts, args):
    stats = dict()
    for opt in opts:
        method = os.path.split(opt.Eval.pred_root)[-1]
        stats[method] = dict()
        for dataset in opt.Eval.datasets:
            stat = pickle.load(open(os.path.join(opt.Eval.pred_root, 'stat', dataset + '.pkl'), 'rb'))
            stats[method][dataset] = stat

    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=len(args.datasets))

    for i, dataset in enumerate(args.datasets):
        for opt in opts:
            method = os.path.split(opt.Eval.pred_root)[-1]
            axes[0][i].plot(stats[method][dataset]['Recall'], stats[method][dataset]['Pre'],    label=method)
            axes[1][i].plot(np.linspace(0, 255, 256), stats[method][dataset]['Fmeasure_Curve'], label=method)
            
    plt.savefig('Figure.png')
            
if __name__ == "__main__":
    args = _args()
    
    opts = []
    for config in args.configs:
        opts.append(load_config(config))
    draw_figure(opts, args)
