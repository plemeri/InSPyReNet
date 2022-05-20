import os
import argparse
import gdown

def gdrive_download(key, filename):
    # os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id={}\" > /dev/null".format(key))
    # os.system("curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id={}\" -o {}".format(key, filename))
    # os.system("rm cookie")
    gdown.download("https://drive.google.com/uc?export=download&id={}".format(key), filename, quiet=False)
    
def unzip(file, dest):
    os.system("unzip {} -d {}".format(file, dest))
    os.system("rm {}".format(file))

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', action='store_true', default=False)
    return parser.parse_args()

download_list = {'dataset':  {'prompt': 'Download RGB SOD Datasets [y/n]: ',                                           
                              'filename': 'RGB_Dataset.zip',   
                              'dest': 'data/RGB_Dataset',   
                              'key': '1kVR8uvjFFqR4Tx3v8XFH6Qp7ugdnBLiG'},
                'backbone':  {'prompt': 'Download ImageNet pre-trained backbone checkpoints [y/n]: ',                  
                              'filename': 'backbone_ckpt.zip', 
                              'dest': 'data/backbone_ckpt', 
                              'key': '1ZtBmUskX5Jmcr1ltlmfJCnwJ7JxjXG4i'},
                'our_ckpt':  {'prompt': 'Download Our pre-trained checkpoints and pre-computed saliency maps [y/n]: ', 
                              'filename': 'snapshots.zip',     
                              'dest': 'snapshots',          
                              'key': '1iD4ekldcivjMJ3gcenW3_kit7TCTMg_S'},
                'sota_ckpt': {'prompt': 'Download SotA checkpoints and pre-computed saliency maps [y/n]: ',            
                              'filename': 'SotA.zip',          
                              'dest': 'snapshots/SotA',     
                              'key': '1X0o7O-dyoLhXvncYpa4pvL6TCe171NXK'}}
    
if __name__ == '__main__':
    args = _args()

    for key in download_list.keys():
        while True:
            dinfo = download_list[key]
            if args.y is False:
                prompt = input(dinfo['prompt']).lower()
            else:
                prompt = 'y'
                
            if prompt in ['y', 'yes']:
                gdrive_download(dinfo['key'], dinfo['filename'])
                unzip(dinfo['filename'], dinfo['dest'])
                break
            elif prompt in ['n', 'no']:
                break
            else:
                print("Bad Input")

    while True:
        if args.y is False:
            prompt = input('Create conda environment [y/n]: ').lower()
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