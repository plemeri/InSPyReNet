import os
import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.misc import *


cfg =   {'DUTS-TE':  [''], 
        'DUT-OMRON': [''], 
        'ECSSD':     [''], 
        'HKU-IS':    [''], 
        'PASCAL-S':  [''],}

def draw_figure(methods, datasets):
    outs = []
    for dataset in datasets:
        imlist = os.listdir(os.path.join('data', 'RGB_Dataset', 'Test_Dataset', dataset, 'masks'))
        # imlist = os.listdir(os.path.join('snapshots', 'SotA', 'PoolNet', dataset))
        # targets = cfg[dataset]
        stats = []
        for method in methods:
            stat = pickle.load(open(os.path.join(method, 'stat', dataset + '.pkl'), 'rb'))['score']
            stats.append(stat)
            print(stat.shape, method)
        stats = np.stack(stats)
        print(stats)
        
        for target in targets:
            out = [cv2.resize(cv2.imread(os.path.join('data', 'RGB_Dataset', 'Test_Dataset', dataset, 'masks', target)), (160, 200))]
            out.append(np.ones((160, 10, 3)) * 255)
            for mdir in methods:
                print(mdir)
                img = cv2.imread(os.path.join(mdir, dataset, target))
                if img is None:
                    print(mdir, dataset, target)
                img = cv2.resize(img, (160, 200))
                out.append(img)
                out.append(np.ones((160, 10, 3)) * 255)
            outs.append(np.hstack(out))
            outs.append(np.ones((10, 210 * (len(methods) + 1), 3)) * 255)
    for i in outs:
        print(i.shape)
    cv2.imwrite('Figure5.png', np.vstack(outs))
            
if __name__ == "__main__":
    theirs = ['PoolNet', 'BASNet', 'EGNet', 'CPD', 'MINet', 'F3Net', 'GateNet', 'LDF', 'PA_KRN', 'VST', 'TTSOD']
    theirs = [os.path.join('snapshots', 'SotA', i) for i in theirs]
    
    ours = ['InSPyReNet_Res2Net50', 'InSPyReNet_Res2Net101', 'InSPyReNet_SwinT', 'InSPyReNet_SwinS', 'InSPyReNet_SwinB']#, 'InSPyReNet_SwinL']
    ours = [os.path.join('snapshots', i) for i in ours]

    methods = ours[::-1] + theirs

    datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    
    draw_figure(methods, datasets)
