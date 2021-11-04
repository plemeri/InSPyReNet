import os
import pandas as pd
import numpy as np

root = 'results'
datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
metrics = {'Sm': 'max', 'avgEm': 'max', 'maxFm': 'max', 'mae': 'min'}

cnn = ['PoolNet', 'PFAN', 'BASNet', 'EGNet', 'CPD', 'MINet', 'F3Net', 'GateNet', 'LDF', 'PA_KRN', 'InSPyReNet_Res2Net50', 'InSPyReNet_Res2Net101']
trn = ['VST', 'TTSOD', 'InSPyReNet_SwinT', 'InSPyReNet_SwinS', 'InSPyReNet_SwinB',  'InSPyReNet_SwinL']

dat = []

for dataset in datasets:
    out = ''
    pkl = os.path.join(root, 'result_' + dataset + '.pkl')
    df = pd.read_pickle(pkl)
    
    dat.append(df[metrics.keys()])
        
merged = pd.concat(dat, axis=1)

cnn_tab = merged.loc[cnn]
trn_tab = merged.loc[trn]

cnn_min = cnn_tab.to_numpy().argsort(axis=0)
cnn_max = len(cnn) - cnn_tab.to_numpy().argsort(axis=0) - 1

trn_min = trn_tab.to_numpy().argsort(axis=0)
trn_max = len(trn) - trn_tab.to_numpy().argsort(axis=0) - 1

out = ''

for i, cnn_ent in enumerate(cnn_tab.iterrows()):
    method, vals = cnn_ent
    out += method
    for j, (key, value) in enumerate(zip(vals.index, vals.values)):
        # print(key, value, cnn_min[i, j], cnn_max[i, j])
        if metrics[key] == 'max':
            order = cnn_max[i, j]
        else:
            order = cnn_min[i, j]
            
        if order == 0:
            color = 'red'
        elif order == 1:
            color = 'blue'
        elif order == 2:
            color = 'green'
        else:
            color = None
        
        if color is None:    
            out += ' && {:.3f}'.format(round(value, 3))
        else:
            out += ' && \\textcolor{{{}}}{{{:.3f}}}'.format(color, round(value, 3))
    out += '\n'
print(out)