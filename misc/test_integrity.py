import os
import argparse
import tqdm
import sys

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.misc import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    return parser.parse_args()

if __name__ == '__main__':
    args = _args()
    opt = load_config(args.config)
    
    gt_root = opt.Eval.gt_root
    pred_root = opt.Eval.pred_root
    datasets = opt.Eval.datasets
    
    integrity = dict()
    
    for dataset in datasets:
        integrity[dataset] = {'length': 'fail', 'name': 'fail', 'note': ''}
        
        pred_db_root = os.path.join(pred_root, dataset)
        preds = os.listdir(pred_db_root)
        
        gt_db_root = os.path.join(gt_root, dataset, 'masks')
        gts = os.listdir(gt_db_root)
        
        if len(preds) == len(gts):
            integrity[dataset]['length'] = 'pass'
        
        preds_sorted = sort([i for i in preds if i in gts])
        gts_sorted =   sort([i for i in gts if i in preds])
        
        if len(preds_sorted) == len(gts_sorted) == len(gts) == len(preds):
            integrity[dataset]['name'] = 'pass'
        else:
            missing = []
            extra = []
            
            for i in preds:
                if i not in gts:
                    extra.append(i)
            
            for i in gts:
                if i not in preds:
                    missing.append(i)
                    
            integrity[dataset]['note'] = ' '.join(['\nmissing[', str(len(missing)), ']:', *missing[:5], '\nextra:[', str(len(extra)), ']:', *extra[:5]])
            
        mismatch = []
        for pred, gt in zip(preds_sorted, gts_sorted):
            pred_ = Image.open(os.path.join(pred_db_root, pred))
            gt_ = Image.open(os.path.join(gt_db_root, gt))
            
            if pred_.size != gt_.size:
                mismatch.append(pred)
                if pred == gt:
                    print('AutoCorrecting size mismatch')
                    pred_ = pred_.resize(gt_.size)
                    pred_.save(os.path.join(pred_db_root, pred))
        
        if len(mismatch) > 0:
            integrity[dataset]['note'] += ' '.join(['\nmismatch[', str(len(mismatch)), ']:', *mismatch[:5]])
                
            
    for key in integrity.keys():
        print(key, '\nlength:', integrity[key]['length'], '\nname:', integrity[key]['name'], integrity[key]['note'], '\n')