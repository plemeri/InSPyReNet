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
    root = 'temp'
    datasets = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    
    method = 'UCNet'
    append  =  {'DUTS-TE':   {'Sm': 0.888, 'mae': 0.034, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.927, 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.860, 'wFm': -np.inf },
                'DUT-OMRON': {'Sm': 0.839, 'mae': 0.051, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.869, 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.773, 'wFm': -np.inf },
                'ECSSD':     {'Sm': 0.921, 'mae': 0.035, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.947, 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.926, 'wFm': -np.inf },
                'HKU-IS':    {'Sm': 0.921, 'mae': 0.026, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': 0.957, 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  0.919, 'wFm': -np.inf },
                'PASCAL-S':  {'Sm': -np.inf, 'mae': np.inf, 'adpEm': -np.inf , 'maxEm': -np.inf , 'avgEm': -np.inf, 'adpFm': -np.inf , 'maxFm': -np.inf , 'avgFm':  -np.inf, 'wFm': -np.inf }}

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

