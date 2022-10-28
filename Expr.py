import torch.cuda as cuda

from utils.misc import *
from run import *

if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    
    train(opt, args)
    cuda.empty_cache()
    if args.local_rank <= 0:
        test(opt, args)
        evaluate(opt, args)
