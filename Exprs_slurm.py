import os
import argparse
import yaml
import copy
import datetime

from utils.misc import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--exprs', type=int, default=4)
    parser.add_argument('--hyp-tune', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--partition', type=str, default='A5000')
    parser.add_argument('--nodelist', type=str, default='n15')
    return parser.parse_args()

def rep_dict(x, klist, val):
    if len(klist) == 1:
        x[klist[0]] = val
    else:
        rep_dict(x[klist[0]], klist[1:], val)

if __name__ == "__main__":
    args = _args()
    
    if args.resume is not None:
        exp_key = args.resume
    else:
        exp_key = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    exp_name = os.path.splitext(os.path.split(args.config)[-1])[0]
    
    print('Expr key:', exp_key)
    
    opt = load_config(args.config, easy=False)

    os.makedirs('temp', exist_ok=True)
    exp_tab = []
    
    for i in range(args.exprs):
        cfg_name = exp_name + '_expr_' + exp_key + '_' + str(i + 1)
        
        if args.resume is None:
            opt_c = copy.deepcopy(opt)
            ckpt_dir = opt_c['Train']['Checkpoint']['checkpoint_dir']
            opt_c['Train']['Checkpoint']['checkpoint_dir'] = os.path.join(*os.path.split(ckpt_dir)[:-1], cfg_name)
            opt_c['Test']['Checkpoint']['checkpoint_dir'] = os.path.join(*os.path.split(ckpt_dir)[:-1], cfg_name)
            opt_c['Eval']['pred_root'] = os.path.join(*os.path.split(ckpt_dir)[:-1], cfg_name)
            
            if args.hyp_tune is True:
                rep_dict(opt_c, opt_c['Train']['HyperTune']['HyperParameter'], 
                        np.linspace(*opt_c['Train']['HyperTune']['Range'])[i % opt_c['Train']['HyperTune']['Range'][-1]].item())

            yaml.dump(opt_c, open(os.path.join('temp', cfg_name + '.yaml'), 'w'), sort_keys=False)
        exp_tab.append((cfg_name, None))

    
    for exp in exp_tab:
        command = 'sbatch ' 
        command += '--partition ' + args.partition + ' '
        command += '--nodelist ' + args.nodelist + ' '
            
        command += ' slurm-submit.sh {}'.format(os.path.join('temp', exp[0] + '.yaml'))
        os.system(command)
        # print(command)