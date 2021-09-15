import os
import argparse
import yaml
import copy
import random
import string

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--exprs', type=int, default=4)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    exp_key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))    
    exp_name = os.path.splitext(os.path.split(args.config)[-1])[0]
    devices = args.devices.split(',')
    opt = yaml.load(open(args.config), yaml.FullLoader)

    os.makedirs('temp', exist_ok=True)
    exp_tab = []

    for i in range(args.exprs):
        opt_c = copy.deepcopy(opt)
        cfg_name = exp_name + '_expr_' + exp_key + '_' + str(i + 1)

        ckpt_dir = opt_c['Train']['Checkpoint']['checkpoint_dir']
        opt_c['Train']['Checkpoint']['checkpoint_dir'] = os.path.join(*os.path.split(ckpt_dir)[:-1], cfg_name)
        opt_c['Test']['Checkpoint']['checkpoint_dir'] = os.path.join(*os.path.split(ckpt_dir)[:-1], cfg_name)
        opt_c['Eval']['pred_root'] = os.path.join(*os.path.split(ckpt_dir)[:-1], cfg_name)

        yaml.dump(opt_c, open(os.path.join('temp', cfg_name + '.yaml'), 'w'), sort_keys=False)
        exp_tab.append((cfg_name, devices[i % len(devices)]))

    for device in devices:
        command = 'tmux new-window \"source ~/.zshrc && conda activate inspyrenet ' 
        for exp in exp_tab:
            if exp[1] == device:
                command += '&& CUDA_VISIBLE_DEVICES={} python Expr.py --config {} '.format(device, os.path.join('temp', exp[0] + '.yaml'))
                if args.verbose is True:
                    command += '--verbose '
        command += '\"'
        os.system(command)