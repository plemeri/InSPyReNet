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


cfg =   {'DUTS-TE':  [], 
        'DUT-OMRON': [], 
        'ECSSD':     ['0001.png', '0760.png'], 
        'HKU-IS':    ['0247.png'], 
        'PASCAL-S':  ['355.png', '25.png'],
        'UHRSD-TE': [
                     '1221.png',
                     '1267.png',
                     '1460.png',
                     '1812.png',
                     '2070.png',
                     '3542.png',
                     '4129.png',
                     '4446.png',
                     '5196.png',
                     '5933.png',
                     '5900.png'
                     ]}

def draw_figure(methods, datasets):
    outs = []
    for dataset in datasets:
        imlist = os.listdir(os.path.join('data', 'RGB_Dataset', 'Test_Dataset', dataset, 'masks'))
        # imlist = os.listdir(os.path.join('snapshots', 'SotA', 'PoolNet', dataset))
        # targets = cfg[dataset]
        
        # stats = []
        # for method in methods:
        #     stat = pickle.load(open(os.path.join(method, 'stat', dataset + '.pkl'), 'rb'))['score']
        #     print(method)
        #     stats.append(stat)
        #     # print(stat.shape, method)
        # stats = np.stack(stats)
        # stats = np.argsort(stats, axis=0)
        # # print(stats)
        # # stats = stats[4] #.sum(axis=0)
        # # stats = stats.argsort()
        
        # ours = stats[:5].mean(axis=0)
        # theirs = stats[5:].mean(axis=0)
        
        # ours = stats[4]
        # theirs = stats[-1]
        
        # stats = ours - theirs
        # score = stats.argsort()[::-1]
        # print(stats[score[:5]])
                
        # targets = np.random.choice(imlist, 1)
        # targets = [imlist[i] for i in score[10:15]]
        # targets = [imlist[stats[-1]]]
        targets = cfg[dataset]
        
        for target in targets:
            # print(target.split('.')[0] + '.jpg')
            print(target)
            out = cv2.resize(cv2.imread(os.path.join('data', 'RGB_Dataset', 'Test_Dataset', dataset, 'images', target if dataset == 'HKU-IS' else target.split('.')[0] + '.jpg')), (200, 160))
            print(out.shape)
            out =  [np.concatenate([out, np.ones((160, 200, 1)) * 255], axis=-1)]
            out.append(np.concatenate([np.ones((160, 10, 3)), np.zeros((160, 10, 1))], axis=-1) * 255)
            out += [np.concatenate([cv2.resize(cv2.imread(os.path.join('data', 'RGB_Dataset', 'Test_Dataset', dataset, 'masks', target)), (200, 160)), np.ones((160, 200, 1)) * 255], axis=-1)]
            out.append(np.concatenate([np.ones((160, 10, 3)), np.zeros((160, 10, 1))], axis=-1) * 255)
            for mdir in methods:
                # print(mdir)
                img = cv2.imread(os.path.join(mdir, dataset, target))
                # if img is None:
                    # print(mdir, dataset, target)
                img = np.concatenate([cv2.resize(img, (200, 160)), np.ones((160, 200, 1)) * 255], axis=-1)
                out.append(img)
                out.append(np.concatenate([np.ones((160, 10, 3)), np.zeros((160, 10, 1))], axis=-1) * 255)
            del out[-1]
            
            outs.append(np.hstack(out))
            outs.append(np.concatenate([np.ones((10, 210 * (len(methods) + 2) - 10, 3)), np.zeros((10, 210 * (len(methods) + 2) - 10, 1))], axis=-1) * 255)
            
    del outs[-1]
    # for i in outs:
        # print(i.shape)
    cv2.imwrite('Figure5.png', np.vstack(outs))
            
if __name__ == "__main__":
    theirs = ['F3Net_SwinB', 'MINet_SwinB', 'LDF_SwinB', 'PA_KRN_SwinB', 'PGNet', 'PGNet_H', 'PGNet_HU'] #['PoolNet', 'BASNet', 'EGNet', 'CPD', 'GateNet', 'F3Net_Res2Net50', 'MINet_Res2Net50', 'LDF_Res2Net50', 'PA_KRN_Res2Net50'] #, ['F3Net_SwinB', 'MINet_SwinB', 'LDF_SwinB', 'PA_KRN_SwinB', 'VST', 'TTSOD']
    theirs = [os.path.join('snapshots', 'SotA', i) for i in theirs]
    
    ours = ['inspyrenet_wo_sica', 'inspyrenet_w_sica', 'glospyre_wo_sica', 'glospyre_w_sica']#['InSPyReNet_Res2Net50'] #, 'InSPyReNet_SwinB']#, 'InSPyReNet_SwinL']
    ours = [os.path.join('snapshots', i) for i in ours]

    methods = ours + theirs

    # datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    datasets = ['UHRSD-TE']
    
    draw_figure(methods, datasets)
