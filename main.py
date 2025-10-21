import argparse
import numpy as np
import torch
from dataset import COCO, COCO_new
from machine import Wv
from options import Options

def main(args):
    # ===========================================================================================
    #  seed
    # ===========================================================================================
    seed = 500  # 627  99(10.26) 500(2024/1/03)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ===========================================================================================
    #  dataset
    # ===========================================================================================
    if args.data in ['10kgray','10kmid','10khigh']:
        dataset_func = COCO
    elif args.data in ['multi2','full2']:
        dataset_func = COCO_new

    train_loader = torch.utils.data.DataLoader(dataset_func('train',args),batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    data_loaders = (train_loader,val_loader)

    
    Train_Machine = Wv(datasets=data_loaders, args=args)
    Train_Machine.train()


if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='Adversairal visible watermarking.'))
    args = parser.parse_args()

    main(args)