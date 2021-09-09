import os
import argparse
import yaml
import sys, traceback
import pprint

from easydict import EasyDict as ed

from run import *
from utils.utils import send_mail


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/UACANet-L.yaml')
    parser.add_argument('--email', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))

    if args.email is True:
        out_str = '### Expr has been started from ' + os.uname()[1] + ' ###\n\n'
        out_str += '### Expr Config ###\n'
        out_str += pprint.pformat(opt, sort_dicts=False)
        send_mail('taehoon1018@postech.ac.kr', 'Expr - ' + args.config + ' has been started', out_str, 'taehoon1018@postech.ac.kr', 'l6uiuoilVlTwrXMORdhf')

    emsg = '### Expr has been ended from ' + os.uname()[1] + ' ###\n\n'

    try:
        train(opt)
        test(opt)
        evaluate(opt)
        exit_code = 0
        emsg += '### Expr Completed ###'
    except:
        exit_code = -1
        emsg += '### Expr exited with Exception ###\n'
        emsg += 'Error: {}'.format(sys.exc_info()[0])

    send_mail('taehoon1018@postech.ac.kr', 'Expr - ' + args.config + ' has been ended', emsg, 'taehoon1018@postech.ac.kr', 'l6uiuoilVlTwrXMORdhf')
    