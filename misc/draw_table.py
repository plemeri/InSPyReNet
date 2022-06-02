import os
import pandas as pd
import numpy as np

root = 'results'
# root = 'temp'

datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
# metrics = {'Sm': 'max', 'maxEm': 'max', 'maxFm': 'max', 'mae': 'min'}

# datasets = ['DAVIS-S', 'HRSOD', 'UHRSD-TE'] 
# metrics = {'Sm': 'max', 'maxEm': 'max', 'maxFm': 'max', 'mae': 'min', 'mBA': 'max'}

# metrics = {'Sm': 'max', 'maxEm': 'max', 'avgEm': 'max', 'maxFm': 'max', 'avgFm': 'max', 'wFm': 'max', 'mae': 'min'}

# datasets = ['DUTS-TE', 'DUT-OMRON']
metrics = {'Sm': 'max', 'maxFm': 'max', 'mae': 'min'}

fields={
#  'PoolNet':              {'name': 'PoolNet \cite{liu2019simple}                ', 'backbone': 'ResNet50'},   
#  'BASNet':               {'name': 'BASNet \cite{qin2019basnet}                 ', 'backbone': 'ResNet34'},   
#  'EGNet':                {'name': 'EGNet \cite{zhao2019egnet}                  ', 'backbone': 'ResNet50'},   
#  'CPD':                  {'name': 'CPD \cite{wu2019cascaded}                   ', 'backbone': 'ResNet50'},   
#  'GateNet':              {'name': 'GateNet \cite{zhao2020suppress}             ', 'backbone': 'ResNeXt101'}, 

#  'RAS_Res2Net50':        {'name': '$^\dagger$RAS \cite{chen2018reverse}        ', 'backbone': 'Res2Net50'},   
#  'F3Net_Res2Net50':      {'name': '$^\dagger$F$^3$Net \cite{wei2020f3net}      ', 'backbone': 'Res2Net50'}, 
#  'LDF_Res2Net50':        {'name': '$^\dagger$LDF \cite{wei2020label}           ', 'backbone': 'Res2Net50'},  
#  'MINet_Res2Net50':      {'name': '$^\dagger$MINet \cite{pang2020multi}        ', 'backbone': 'Res2Net50'},
#  'PA_KRN_Res2Net50':     {'name': '$^\dagger$PA-KRN \cite{xu2021locate}        ', 'backbone': 'Res2Net50'},   
#  'InSPyReNet_Res2Net50': {'name': '\textit{Ours}                               ', 'backbone': 'Res2Net50'},  
 
 'VST':                  {'name': 'VST \cite{liu2021visual}                    ', 'backbone': 'T2T-ViT-14'}, 
 'TTSOD':                {'name': 'TTSOD \cite{mao2021transformer}             ', 'backbone': 'SwinB$^*$'},
 'RAS_SwinB':            {'name': '$^\dagger$RAS \cite{chen2018reverse}        ', 'backbone': 'SwinB$^*$'},  
 'F3Net_SwinB':          {'name': '$^\dagger$F$^3$Net \cite{wei2020f3net}      ', 'backbone': 'SwinB$^*$'}, 
 'LDF_SwinB':            {'name': '$^\dagger$LDF \cite{wei2020label}           ', 'backbone': 'SwinB$^*$'},  
 'MINet_SwinB':          {'name': '$^\dagger$MINet \cite{pang2020multi}        ', 'backbone': 'SwinB$^*$'},
 'PA_KRN_SwinB':         {'name': '$^\dagger$PA-KRN \cite{xu2021locate}        ', 'backbone': 'SwinB$^*$'},   
 'InSPyReNet_SwinB':     {'name': '\textit{Ours}                               ', 'backbone': 'SwinB$^*$'}
#  'HRSOD':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
#  'DHQSOD':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
#  'PGNet':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
#  'PGNet_H':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
#  'PGNet_HU':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
#  'RAS_SwinB':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
#  'InSPyReNet_SwinB_HU_LR':     {'name': 'Ours                                        ', 'backbone': 'SwinB$^*$'},
 }



dat = []

for dataset in datasets:
    out = ''
    pkl = os.path.join(root, 'result_' + dataset + '.pkl')
    df = pd.read_pickle(pkl)
    
    dat.append(df[metrics.keys()])
        
merged = pd.concat(dat, axis=1)

methods = list(fields.keys())

tab = merged.loc[methods]

min = tab.to_numpy().argsort(axis=0)
max = len(methods) - tab.to_numpy().argsort(axis=0) - 1

out = ''
rv = 3

for i, ent in enumerate(tab.iterrows()):
    method, vals = ent
    head = fields[method]['name']#tablename[i] # method if 'InSPyRe' not in method else 'InSPyReNet (Ours)'
    out +=  head + ' & ' + str(fields[method]['backbone']) # str(backbones[i]) #+ ' & ' + str(macs[i]) + ' & ' + str(params[i])
    # out +=  head + ' & ' + str(macs[i]) + ' & ' + str(params[i])
    for j, (key, value) in enumerate(zip(vals.index, vals.values)):
        # print(key, value, min[i, j], max[i, j])
        if metrics[key] == 'min':
            order = tab.to_numpy()[:, j].round(rv)
            rank = np.sort(np.unique(order))
        else:
            order = tab.to_numpy()[:, j].round(rv)
            rank = np.sort(np.unique(order))[::-1]
            
        color = None
        if order[i] == rank[0]:
            color = 'red'
        elif order[i] == rank[1]:
            color = 'blue'
        # elif order[i] == rank[2]:
        #     color = 'green'
            
        if value == np.inf or value == -np.inf:
            value = '-'
        
        if type(value) == np.float64:
            value = ('{:.'+str(rv)+'f}').format(round(value, rv))
        
        if color is None:    
            out += ' & {}'.format(value)#.replace('0.', '.')
        else:
            out += ' & \\textcolor{{{}}}{{{}}}'.format(color, value)#.replace('0.', '.')
    out += ' \\\\ \n'
print(out)