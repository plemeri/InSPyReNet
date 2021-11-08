import os
import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.misc import *


cfg =   {'DUTS-TE':  {'ylim': ((0.6, 1.0),  (0.6, 0.95))}, 
        'DUT-OMRON': {'ylim': ((0.6, 0.92), (0.6, 0.87))}, 
        'ECSSD':     {'ylim': ((0.92, 1.0), (0.8, 0.97))}, 
        'HKU-IS':    {'ylim': ((0.9, 1.0),  (0.8, 0.96))}, 
        'PASCAL-S':  {'ylim': ((0.7, 0.98),  (0.7, 0.92))}, }

def draw_figure(opts, datasets):
    stats = dict()
    for opt in opts:
        method = os.path.split(opt.Eval.pred_root)[-1]
        stats[method] = dict()
        for dataset in opt.Eval.datasets:
            stat = pickle.load(open(os.path.join(opt.Eval.pred_root, 'stat', dataset + '.pkl'), 'rb'))
            stats[method][dataset] = stat

    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(25, 5))
    axes = [axes]

    for i, dataset in enumerate(datasets):
        axes[0][i].set_xlabel('Recall'   , fontsize=20)
        # axes[1][i].set_xlabel('Threshold', fontsize=15)

        axes[0][i].set_ylabel('Precision', fontsize=20)
        # axes[1][i].set_ylabel('F-measure', fontsize=15)
        
        axes[0][i].set_xlim((0, 1.0))
        # axes[1][i].set_xlim((0, 255))
        
        axes[0][i].set_ylim(*cfg[dataset]['ylim'][0])
        # axes[1][i].set_ylim(*cfg[dataset]['ylim'][1])
        
        axes[0][i].set_yticks(np.linspace(*cfg[dataset]['ylim'][0], 7).round(3))
        axes[0][i].yaxis.set_tick_params(labelsize=12, rotation=40)
        axes[0][i].xaxis.set_tick_params(labelsize=12)
        # axes[1][i].set_yticks(np.linspace(*cfg[dataset]['ylim'][1], 7).round(3))
        axes[0][i].set_title(dataset, fontsize=20, fontweight='bold')
        axes[0][i].grid()
        # axes[1][i].grid()
        
        # lines, labels = axes[0][i].get_legend_handles_labels()
        
        # axes[0][i].legend()
        
        for opt in opts:
            method = os.path.split(opt.Eval.pred_root)[-1]
            axes[0][i].plot(stats[method][dataset]['Recall'], stats[method][dataset]['Pre'],    '--' if 'InSPyRe' not in method else '-', label=method, linewidth=2)
            # axes[1][i].plot(np.linspace(0, 255, 256), stats[method][dataset]['Fmeasure_Curve'], '--' if 'InSPyRe' not in method else '-', label=method, )
        axes[0][i].legend(loc = 'lower left')
    
    # lines, labels = axes[0][-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc = 'upper center')
    plt.tight_layout()
    plt.savefig('Figure3.png', transparent=True)
    # plt.savefig('Figure3.png')
            
if __name__ == "__main__":
    theirs = ['PoolNet', 'BASNet', 'EGNet', 'CPD', 'MINet', 'F3Net', 'GateNet', 'LDF', 'PA_KRN', 'VST', 'TTSOD']
    theirs = [os.path.join('configs', 'SotA', i + '.yaml') for i in theirs]
    
    ours = ['InSPyReNet_Res2Net50', 'InSPyReNet_Res2Net101', 'InSPyReNet_SwinT', 'InSPyReNet_SwinS', 'InSPyReNet_SwinB']#, 'InSPyReNet_SwinL']
    ours = [os.path.join('configs', i + '.yaml') for i in ours]

    configs = theirs + ours

    datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    
    opts = []
    for config in configs:
        opts.append(load_config(config))
    draw_figure(opts, datasets)
