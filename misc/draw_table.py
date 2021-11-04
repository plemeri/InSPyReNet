import os
import pandas as pd
import numpy as np

# root = 'results'
root = 'temp'
datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
metrics = {'Sm': 'max', 'avgEm': 'max', 'maxFm': 'max', 'mae': 'min'}

cnn = ['PoolNet', 'PFAN', 'BASNet', 'EGNet', 'CPD', 'MINet', 'F3Net', 'GateNet', 'LDF', 'UCNet', 'PA_KRN', 'InSPyReNet_Res2Net50', 'InSPyReNet_Res2Net101']
cnn_backbones = ['ResNet-50', '']
cnn_macs = [128.4, 33.5, 286.6, 350.2, 21.1, 125.3, 19.6, 162.1, 18.5, '-', 256.5, 59.4, 68.8]
cnn_params = [69.5, 16.4, 87.1, 111.7, 47.9 , 162.4, 25.5, 128.6, 25.2, '-', 141.1, 28.1, 47.6]

trn = ['VST', 'TTSOD', 'InSPyReNet_SwinT', 'InSPyReNet_SwinS', 'InSPyReNet_SwinB',  'InSPyReNet_SwinL']
trn_backbones = []
trn_macs =   [23.2, '-', 70.4, 84.1, 101.3, 157.4]
trn_params = [44.0, '-', 31.1, 52.4, 90.5, 199.0]

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
    out += method + ' && ' + str(cnn_macs[i]) + ' && ' + str(cnn_params[i]) + ' && '
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
            
        if value == np.inf or value == -np.inf:
            value = '-'
        
        if type(value) == np.float64:
            value = '{:.3f}'.format(round(value, 3))
        
        if color is None:    
            out += ' && {}'.format(value)
        else:
            out += ' && \\textcolor{{{}}}{{{}}}'.format(color, value)
    out += '\n'
print(out)