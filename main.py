'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-10 11:09:27
LastEditors: Cai Weichao
LastEditTime: 2022-03-16 22:19:17
'''
from utils.parser import BaseParser
import numpy as np
import torch
import numpy as np
from train import TrainProcessor
import random


def main(args):
    # set seed
    seed = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # train         
    train = TrainProcessor(args=args)
    train.txt_logger.info('Train Start') 
    
    #train.train()


if __name__ == '__main__':
    parser = BaseParser() # add other argument is in utils->parser.py
    args = parser.add_other_argument()
    main(args)
    
