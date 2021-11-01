#!/bin/bash
import os
import argparse

def gdrive_download(key, filename):
    os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id={}}\" > /dev/null".format(key))
    os.system("curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id={}}\" -o {}}".format(key, filename))
    os.system("rm cookie")
    
def unzip(file, dest):
    os.system("unzip {} -d {}".format(file, dest))
    os.system("rm {}".format(file))

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', action='store_true', default=False)
    return parser.parse_args()

download_list = {'dataset':   {'filename': 'RGB_Dataset.zip',   'dest': 'data/RGB_Dataset',   'key': '1QKYG58WLDdq4ar690H5qCNiSgI1twKie'},
                 'backbone':  {'filename': 'backbone_ckpt.zip', 'dest': 'data/backbone_ckpt', 'key': '1B53u3WApaicCsaLYZIKjNSrmhe8qKeGP'},
                 'our_ckpt':  {'filename': 'snapshots.zip',     'dest': 'snapshots',          'key': '1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0'},
                 'sota_ckpt': {'filename': 'SotA.zip',          'dest': 'snapshots/SotA',     'key': '1eBDsPD_iPj_skILTXutyAJ4hjmSZuyAb'}}
    
if __name__ == '__main__':
    args = _args()

    while True:
        if args.y is False:
            prompt = input('Download RGB SOD Datasets [y/n]: ').lower()
        else:
            prompt = 'y'
            
        if prompt in ['y', 'yes']:
            dinfo = download_list['dataset']
            gdrive_download(dinfo['key'], dinfo['filename'])
            unzip(dinfo['filename'], dinfo['dest'])
            # os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=1QKYG58WLDdq4ar690H5qCNiSgI1twKie\" > /dev/null")
            # os.system("curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1QKYG58WLDdq4ar690H5qCNiSgI1twKie\" -o RGB_Dataset.zip")
            # os.system("rm cookie")
            # os.system("unzip RGB_Dataset.zip -d data/RGB_Dataset")
            # os.system("rm RGB_Dataset.zip")
            break
        elif prompt in ['n', 'no']:
            break
        else:
            print("Bad Input")


    while True:
        if args.y is False:
            prompt = input('Download ImageNet pre-trained backbone checkpoints [y/n]: ').lower()
        else:
            prompt = 'y'
            
        if prompt in ['y', 'yes']:
            dinfo = download_list['backbone']
            gdrive_download(dinfo['key'], dinfo['filename'])
            unzip(dinfo['filename'], dinfo['dest'])
            # os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=1B53u3WApaicCsaLYZIKjNSrmhe8qKeGP\" > /dev/null")
            # os.system("curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1B53u3WApaicCsaLYZIKjNSrmhe8qKeGP\" -o backbone_ckpt.zip")
            # os.system("rm cookie")
            # os.system("unzip backbone_ckpt.zip -d data/backbone_ckpt")
            # os.system("rm backbone_ckpt.zip")
            break
        elif prompt in ['n', 'no']:
            break
        else:
            print("Bad Input")

    while True:
        if args.y is False:
            prompt = input('Download Our pre-trained checkpoints and pre-computed saliency maps [y/n]: ').lower()
        else:
            prompt = 'y'
            
        if prompt in ['y', 'yes']:
            
            dinfo = download_list['our_ckpt']
            gdrive_download(dinfo['key'], dinfo['filename'])
            unzip(dinfo['filename'], dinfo['dest'])
            # os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0\" > /dev/null")
            # os.system("curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0\" -o snapshots.zip")
            # os.system("rm cookie")
            # os.system("unzip snapshots.zip -d snapshots")
            # os.system("rm snapshots.zip")
            break
        elif prompt in ['n', 'no']:
            break
        else:
            print("Bad Input")

    while True:
        if args.y is False:
            prompt = input('Download SotA checkpoints and pre-computed saliency maps [y/n]: ').lower()
        else:
            prompt = 'y'
            
        if prompt in ['y', 'yes']:
            dinfo = download_list['sota_ckpt']
            gdrive_download(dinfo['key'], dinfo['filename'])
            unzip(dinfo['filename'], dinfo['dest'])
            # os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0\" > /dev/null")
            # os.system("curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0\" -o SotA.zip")
            # os.system("rm cookie")
            # os.system("unzip SotA.zip -d snapshots/SotA")
            # os.system("rm SotA.zip")
            break
        elif prompt in ['n', 'no']:
            break
        else:
            print("Bad Input")

    while True:
        if args.y is False:
            prompt = input('Download Our pre-trained checkpoints [y/n]: ').lower()
        else:
            prompt = 'y'
            
        if prompt in ['y', 'yes']:
            os.system("source ~/.zshrc")
            os.system("conda create -y -n inspyrenet python=3.8")
            os.system("\"${HOME}/anaconda3/envs/inspyrenet/bin/python\" -m pip install -r requirements.txt")
        elif prompt in ['n', 'no']:
            break
        else:
            print("Bad Input")