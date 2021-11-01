import os
import argparse
import tqdm
import sys
import pickle

import numpy as np

from PIL import Image
from tabulate import tabulate

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.eval_functions import *
from utils.utils import *

BETA = 1.0


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--save-stat', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser.parse_args()


def evaluate(opt, args):
    if os.path.isdir(opt.Eval.result_path) is False:
        os.makedirs(opt.Eval.result_path)
        
    if args.save_stat is True and os.path.isdir(os.path.join(opt.Eval.pred_root, 'stat')) is False:
        os.makedirs(os.path.join(opt.Eval.pred_root, 'stat'))

    method = os.path.split(opt.Eval.pred_root)[-1]
    Thresholds = np.linspace(1, 0, 256)
    headers = opt.Eval.metrics
    results = []

    if args.verbose is True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(opt.Eval.datasets, desc='Expr - ' + method, total=len(
            opt.Eval.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = opt.Eval.datasets

    for dataset in datasets:
        pred_root = os.path.join(opt.Eval.pred_root, dataset)
        gt_root = os.path.join(opt.Eval.gt_root, dataset, 'masks')

        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        hitRate = np.zeros((len(preds), len(Thresholds)))
        falseAlarm = np.zeros((len(preds), len(Thresholds)))

        IoU = np.zeros((len(preds), len(Thresholds)))
        TPR = np.zeros((len(preds), len(Thresholds)))
        FPR = np.zeros((len(preds), len(Thresholds)))
        Pre = np.zeros((len(preds), len(Thresholds)))
        Recall = np.zeros((len(preds), len(Thresholds)))

        MAE = np.zeros(len(preds))
        S_measure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        Emeasure = np.zeros(len(preds))

        if args.verbose is True:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
                preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample

            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))

            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            gtSize = gt_mask.shape

            assert pred_mask.shape == gt_mask.shape

            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(
                np.float64) / (pred_mask.max() + np.finfo(np.float64).eps)

            IoU[i], TPR[i], FPR[i], Pre[i], Recall[i], hitRate[i], falseAlarm[i] = thresholdBased_HR_FR(
                pred_mask, Thresholds, gt_mask)  # index 0 is not working

            S_measure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = wFmeasure_calu(pred_mask, gt_mask)
            MAE[i] = CalMAE(pred_mask, gt_mask)
            Emeasure[i] = Emeasure_calu(pred_mask, gt_mask)
        result = []

        mae = np.nanmean(MAE)
        Sm = np.nanmean(S_measure)
        wFm = np.nanmean(wFmeasure)
        Em = np.nanmean(Emeasure)

        Pre = np.nanmean(Pre, axis=0)
        Recall = np.nanmean(Recall, axis=0)
        IoU = np.nanmean(IoU, axis=0)
        TPR = np.nanmean(TPR, axis=0)
        hitRate = np.nanmean(hitRate, axis=0)
        falseAlarm = np.nanmean(falseAlarm, axis=0)

        Fmeasure_Curve = 1.3 * Pre * Recall / (0.3 * Pre + Recall)
        maxF = np.max(Fmeasure_Curve)
        avgF = np.mean(Fmeasure_Curve)
        IoUmaxF = IoU[np.argmax(Fmeasure_Curve)]
        maxIoU = np.max(IoU)
        meanIoU = np.mean(IoU)

        out = []
        for metric in opt.Eval.metrics:
            out.append(eval(metric))

        result.extend(out)
        results.append([dataset, *result])

        csv = os.path.join(opt.Eval.result_path, 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(['method', *headers]) + '\n')

        out_str = method + ','
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'

        csv.write(out_str)
        csv.close()
        
        if args.save_stat is True:
            stat = {'Pre': Pre, 'Recall': Recall, 'Fmeasure_Curve': Fmeasure_Curve}
            with open(os.path.join(opt.Eval.pred_root, 'stat', dataset + '.pkl'), 'wb') as f:
                pickle.dump(stat, f)

    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")
    
    if args.verbose is True:
        print(tab)
        print("#"*20, "End Evaluation", "#"*20)

    return tab


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    evaluate(opt, args)
