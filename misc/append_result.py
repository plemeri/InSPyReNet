import os
import argparse
import tqdm
import sys
import pickle

import pandas as pd
import numpy as np

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.eval_functions import *
from utils.misc import *

    
if __name__ == "__main__":
    root = 'results'
    datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    
    # method = 'UCNet'
    # append  =  {'DUTS-TE':   {'Sm': 0.888  , 'mae': 0.034 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.927  , 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.860  , 'wFm': -np.inf },
    #             'DUT-OMRON': {'Sm': 0.839  , 'mae': 0.051 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.869  , 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.773  , 'wFm': -np.inf },
    #             'ECSSD':     {'Sm': 0.921  , 'mae': 0.035 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.947  , 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.926  , 'wFm': -np.inf },
    #             'HKU-IS':    {'Sm': 0.921  , 'mae': 0.026 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.957  , 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.919  , 'wFm': -np.inf },
    #             'PASCAL-S':  {'Sm': -np.inf, 'mae': np.inf, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  -np.inf, 'wFm': -np.inf }}

    # method = 'ABiUNet'
    # append  =  {'DUTS-TE':   {'Sm': 0.904  , 'mae': 0.029 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.906, 'avgFm':  0.860, 'wFm': -np.inf },
    #             'DUT-OMRON': {'Sm': 0.860  , 'mae': 0.043 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.843, 'avgFm':  0.773, 'wFm': -np.inf },
    #             'ECSSD':     {'Sm': 0.936  , 'mae': 0.026 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.959, 'avgFm':  0.926, 'wFm': -np.inf },
    #             'HKU-IS':    {'Sm': 0.932  , 'mae': 0.021 , 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.951, 'avgFm':  0.919, 'wFm': -np.inf },
    #             'PASCAL-S':  {'Sm': -np.inf, 'mae': np.inf, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': -np.inf, 'avgFm':  -np.inf, 'wFm': -np.inf }}
    
    # method = 'RFBB'
    # append  =  {'DUTS-TE':   {'Sm': 0.910  , 'mae': 0.025 , 'adpEm': -np.inf , 'maxEm': 0.925 , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.929, 'avgFm':  0.890, 'wFm': -np.inf },
    #             'DUT-OMRON': {'Sm': 0.847  , 'mae': 0.040 , 'adpEm': -np.inf , 'maxEm': 0.878 , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.838, 'avgFm':  0.804, 'wFm': -np.inf },
    #             'ECSSD':     {'Sm': 0.941  , 'mae': 0.022 , 'adpEm': -np.inf , 'maxEm': 0.932 , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.964, 'avgFm':  0.949, 'wFm': -np.inf },
    #             'HKU-IS':    {'Sm': 0.933  , 'mae': 0.020 , 'adpEm': -np.inf , 'maxEm': 0.965 , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.953, 'avgFm':  0.936, 'wFm': -np.inf },
    #             'PASCAL-S':  {'Sm': 0.867  , 'mae': 0.050 , 'adpEm': -np.inf , 'maxEm': 0.866 , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': 0.890, 'avgFm':  0.862, 'wFm': -np.inf }}

    results = []
    for dataset in append.keys():
        out = append[dataset]
        print(out)

        pkl = os.path.join(root, 'result_' + dataset + '.pkl')
        if os.path.isfile(pkl) is True:
            result = pd.read_pickle(pkl)
            result.loc[method] = out
            result.to_pickle(pkl)
        else:
            result = pd.DataFrame(data=out, index=[method])
            result.to_pickle(pkl)
        result.to_csv(os.path.join(root, 'result_' + dataset + '.csv'))
        results.append(result)
        
    for dataset, result in zip(datasets, results):
        print('###', dataset, '###', '\n', result.sort_index(), '\n')

