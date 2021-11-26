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

BETA = 1.0


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--stat', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


def evaluate(opt, args):
    if os.path.isdir(opt.Eval.result_path) is False:
        os.makedirs(opt.Eval.result_path)
        
    if args.stat is True and os.path.isdir(os.path.join(opt.Eval.pred_root, 'stat')) is False:
        os.makedirs(os.path.join(opt.Eval.pred_root, 'stat'))

    method = os.path.split(opt.Eval.pred_root)[-1]

    if args.verbose is True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(opt.Eval.datasets, desc='Expr - ' + method, total=len(
            opt.Eval.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = opt.Eval.datasets

    results = []

    for dataset in datasets:
        pred_root = os.path.join(opt.Eval.pred_root, dataset)
        gt_root = os.path.join(opt.Eval.gt_root, dataset, 'masks')

        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)

        preds = sort(preds)
        gts = sort(gts)
        
        preds = [i for i in preds if i in gts]
        gts = [i for i in gts if i in preds]
        
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        MAE = Mae()

        if args.verbose is True:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
                preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample

            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)).convert('L'))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)).convert('L'))

            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]

            assert pred_mask.shape == gt_mask.shape, print(pred, 'does not match the size of', gt)
            # print(gt_mask.max())

            FM.step( pred=pred_mask, gt=gt_mask)
            WFM.step(pred=pred_mask, gt=gt_mask)
            SM.step( pred=pred_mask, gt=gt_mask)
            EM.step( pred=pred_mask, gt=gt_mask)
            MAE.step(pred=pred_mask, gt=gt_mask)
            
        result = []

        Sm =  SM.get_results()["sm"]
        wFm = WFM.get_results()["wfm"]
        mae = MAE.get_results()["mae"]
        
        Fm =  FM.get_results()["fm"]
        Em =  EM.get_results()["em"]
        
        adpEm = Em["adp"]
        avgEm = Em["curve"].mean()
        maxEm = Em["curve"].max()
        adpFm = Fm["adp"]
        avgFm = Fm["curve"].mean()
        maxFm = Fm["curve"].max()
        
        out = dict()
        for metric in opt.Eval.metrics:
            out[metric] = eval(metric)

        pkl = os.path.join(opt.Eval.result_path, 'result_' + dataset + '.pkl')
        if os.path.isfile(pkl) is True:
            result = pd.read_pickle(pkl)
            result.loc[method] = out
            result.to_pickle(pkl)
        else:
            result = pd.DataFrame(data=out, index=[method])
            result.to_pickle(pkl)
        result.to_csv(os.path.join(opt.Eval.result_path, 'result_' + dataset + '.csv'))
        
        results.append(result)
        
        if args.stat is True:
            Fm_info = FM.get_results()
            Fm = Fm_info["fm"]
            PR = Fm_info["pr"]
            stat = {'Pre': np.flip(PR["p"]), 'Recall': np.flip(PR["r"]), 'Fmeasure_Curve': np.flip(Fm["curve"]), 'score': np.array(SM.sms)}
            with open(os.path.join(opt.Eval.pred_root, 'stat', dataset + '.pkl'), 'wb') as f:
                pickle.dump(stat, f)
                
    if args.verbose is True:
        for dataset, result in zip(datasets, results):
            print('###', dataset, '###', '\n', result.sort_index(), '\n')
    
if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    evaluate(opt, args)
