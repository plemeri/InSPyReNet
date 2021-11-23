import os
import pandas as pd
import numpy as np

root = 'results'
# root = 'temp'
datasets = ['ECSSD', 'HKU-IS', 'PASCAL-S'] #['DUTS-TE', 'DUT-OMRON']  #['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S'] 
# metrics = {'Sm': 'max', 'maxEm': 'max', 'maxFm': 'max', 'mae': 'min'}
metrics = {'Sm': 'max', 'maxEm': 'max', 'avgEm': 'max', 'maxFm': 'max', 'avgFm': 'max', 'wFm': 'max', 'mae': 'min'}

# methods =   ['PoolNet',   'BASNet',    'EGNet',     'CPD',       'MINet',     'F3Net',     'GateNet',     'LDF',       'UCNet',     'PA_KRN',    'VST',        'ABiUNet',   'TTSOD',     'RFBB',             'InSPyReNet_Res2Net50', 'InSPyReNet_Res2Net101', 'InSPyReNet_SwinT', 'InSPyReNet_SwinS', 'InSPyReNet_SwinB']#, 'InSPyReNet_SwinL']
# tablename = [
# 'PoolNet \cite{liu2019simple}                ',
# 'BASNet \cite{qin2019basnet}                 ',
# 'EGNet \cite{zhao2019egnet}                  ',
# 'CPD \cite{wu2019cascaded}                   ',
# 'MINet \cite{pang2020multi}                  ',
# 'F$^3$Net \cite{wei2020f3net}                ',
# 'GateNet \cite{zhao2020suppress}             ',
# 'LDF \cite{wei2020label}                     ',
# 'UCNet$^\dagger$ \cite{zhang2021uncertainty} ',
# 'PA-KRN \cite{xu2021locate}                  ',
# 'VST \cite{liu2021visual}                    ',
# 'ABiUNet$^\dagger$ \cite{qiu2021boosting}    ',
# 'TTSOD \cite{mao2021transformer}             ',
# 'RFBB$^\dagger$ \cite{ma2021receptive}       ',
# 'Ours                                        ',
# 'Ours                                        ',
# 'Ours                                        ',
# 'Ours                                        ',
# 'Ours                                        '
# ]
# backbones = ['ResNet50', 'ResNet34', 'ResNet50', 'ResNet50', 'ResNet50', 'ResNet50', 'ResNeXt101', 'ResNet50', 'ResNet50', 'ResNet50', 'T2T-ViT-14', 'PVT'    ,   'SwinB$^*$', 'ResNet18 \\& Swin', 'Res2Net50',            'Res2Net101',            'SwinT',            'SwinS',            'SwinB$^*$'       ]#, 'SwinL$^*$']
# macs =      [128.4        ,286.6       ,350.2       ,21.1        ,125.3       ,19.6         ,162.1        ,18.5        ,'-'         ,256.5,       23.2,         '-'      ,   '-',         '-',               59.4                   ,68.8                    ,70.4               ,84.1               ,101.3]#, 157.4]
# params =    [69.5         ,87.1        ,111.7       ,47.9        ,162.4       ,25.5         ,128.6        ,25.2        ,'-'         ,141.1,       44.0,         '35.0'   ,   '-',         '-',               28.1                   ,47.6                    ,31.1               ,52.4               ,90.5 ]#,  199.0]

methods =   ['PoolNet',   'BASNet',    'EGNet',     'CPD',       'MINet',     'F3Net',     'GateNet',     'LDF',       'UCNet',     'PA_KRN',    'VST',        'ABiUNet',   'TTSOD',      'InSPyReNet_Res2Net50', 'InSPyReNet_Res2Net101', 'InSPyReNet_SwinT', 'InSPyReNet_SwinS', 'InSPyReNet_SwinB']#, 'InSPyReNet_SwinL']
tablename = [
'PoolNet \cite{liu2019simple}                ',
'BASNet \cite{qin2019basnet}                 ',
'EGNet \cite{zhao2019egnet}                  ',
'CPD \cite{wu2019cascaded}                   ',
'MINet \cite{pang2020multi}                  ',
'F$^3$Net \cite{wei2020f3net}                ',
'GateNet \cite{zhao2020suppress}             ',
'LDF \cite{wei2020label}                     ',
'UCNet$^\dagger$ \cite{zhang2021uncertainty} ',
'PA-KRN \cite{xu2021locate}                  ',
'VST \cite{liu2021visual}                    ',
'ABiUNet$^\dagger$ \cite{qiu2021boosting}    ',
'TTSOD \cite{mao2021transformer}             ',
'Ours                                        ',
'Ours                                        ',
'Ours                                        ',
'Ours                                        ',
'Ours                                        '
]
backbones = ['ResNet50', 'ResNet34', 'ResNet50', 'ResNet50', 'ResNet50', 'ResNet50', 'ResNeXt101', 'ResNet50', 'ResNet50', 'ResNet50', 'T2T-ViT-14', 'PVT'    ,   'SwinB$^*$',  'Res2Net50',            'Res2Net101',            'SwinT',            'SwinS',            'SwinB$^*$'       ]#, 'SwinL$^*$']
macs =      [128.4        ,286.6       ,350.2       ,21.1        ,125.3       ,19.6         ,162.1        ,18.5        ,'-'         ,256.5,       23.2,         '-'      ,   '-',            59.4                   ,68.8                    ,70.4               ,84.1               ,101.3]#, 157.4]
params =    [69.5         ,87.1        ,111.7       ,47.9        ,162.4       ,25.5         ,128.6        ,25.2        ,'-'         ,141.1,       44.0,         '35.0'   ,   '-',            28.1                   ,47.6                    ,31.1               ,52.4               ,90.5 ]#,  199.0]



dat = []

for dataset in datasets:
    out = ''
    pkl = os.path.join(root, 'result_' + dataset + '.pkl')
    df = pd.read_pickle(pkl)
    
    dat.append(df[metrics.keys()])
        
merged = pd.concat(dat, axis=1)

tab = merged.loc[methods]

min = tab.to_numpy().argsort(axis=0)
max = len(methods) - tab.to_numpy().argsort(axis=0) - 1

out = ''
rv = 3

for i, ent in enumerate(tab.iterrows()):
    method, vals = ent
    head = tablename[i] # method if 'InSPyRe' not in method else 'InSPyReNet (Ours)'
    out +=  head + ' & ' + str(backbones[i]) #+ ' & ' + str(macs[i]) + ' & ' + str(params[i])
    for j, (key, value) in enumerate(zip(vals.index, vals.values)):
        # print(key, value, min[i, j], max[i, j])
        if metrics[key] == 'min':
            order = tab.to_numpy()[:, j].round(rv)
            rank = np.sort(np.unique(order))
        else:
            order = tab.to_numpy()[:, j].round(rv)
            rank = np.sort(np.unique(order))[::-1]
            
        if order[i] == rank[0]:
            color = 'red'
        elif order[i] == rank[1]:
            color = 'blue'
        elif order[i] == rank[2]:
            color = 'green'
        else:
            color = None
            
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