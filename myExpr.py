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
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--email', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    account = ed(yaml.load(open('configs/email.yaml'), yaml.FullLoader))

    if args.email is True:
        out_str = '### Expr has been started from ' + os.uname()[1] + ' ###\n\n'
        out_str += '### Expr Config ###\n'
        out_str += pprint.pformat(opt, sort_dicts=False)
        send_mail(account.address, 'Expr - ' + args.config + ' has been started', out_str, account.address, account.password)

    emsg = '### Expr has been ended from ' + os.uname()[1] + ' ###\n\n'

    try:
        train(opt, verbose=args.verbose)
        test(opt, verbose=args.verbose)
        evl = evaluate(opt, verbose=args.verbose)
        exit_code = 0
        emsg += '### Expr Completed ###\n'
        emsg += evl
    except:
        exit_code = -1
        emsg += '### Expr exited with Exception ###\n'
        emsg += 'Error: {}'.format(traceback.format_exc())

    if args.email is True:
        send_mail(account.address, 'Expr - ' + args.config + ' has been ended', emsg, account.address, account.password)
    