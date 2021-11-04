import os
import argparse
import tqdm


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--suffix', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = _args()
    
    root = args.source
    files = os.listdir(root)
    
    for file in tqdm.tqdm(files):
        nfile = file
        if args.suffix in file:
            nfile = file.replace(args.suffix, '')
        os.system(' '.join(['mv', os.path.join(root, file), os.path.join(root, nfile)]))