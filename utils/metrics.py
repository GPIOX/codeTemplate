'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-12 15:14:27
LastEditors: Cai Weichao
LastEditTime: 2022-04-11 19:41:28
'''
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union
import numpy as np

class DiceLoss(object):
    def __init__(self):
        pass

    '''
    name: diceCoefficient
    brief: calculates the dice coefficient
    author: Cai Weichao
    param {Optional} pred: N, C, H, W, and C is the number of class
    param {Optional} label: N, H, W
    param {float} epsilon: avoid smoothing terms with 0 denominator
    return dice coefficient
    '''    
    def diceCoefficient(self, pred: Optional[Tensor], label: Optional[Tensor], epsilon: float=1e-6):
        N = label.size()[0]
        numerator = (pred * label).sum(dim=-1).sum(dim=-1) * 2
        #denominator = pred.pow(2).sum(dim=-1).sum(dim=-1) + label.pow(2).sum(dim=-1).sum(dim=-1) + epsilon # why using pow function?
        denominator = pred.sum(dim=-1).sum(dim=-1) + label.sum(dim=-1).sum(dim=-1) + epsilon
        dice_val = numerator/denominator
        dice_coeff = dice_val.sum() / N
        return dice_coeff

    '''
    name: binaryDiceLoss
    brief: dice loss function for binary classification
    author: Cai Weichao
    param {Optional} pred: N, C, H, W, and C is the number of class
    param {Optional} label: N, H, W
    param {float} epsilon: avoid smoothing terms with 0 denominator
    return dice loss value
    '''    
    def binaryDiceLoss(self, pred: Optional[Tensor], label: Optional[Tensor], epsilon: float=1e-6):
        dice_coeff = self.diceCoefficient(pred, label, epsilon)
        return 1 - dice_coeff

    '''
    name: multiClassDiceLoss
    brief: dice loss function for multi-classification, 
           When there are several categories, One Hot converts Label into multiple binary classification
    author: Cai Weichao
    param {*} self
    param {Optional} pred: N, C, H, W, and C is the number of class
    param {Optional} label: N, H, W, then will change to one-hot
    param {float} epsilon
    return {*}
    '''    
    def multiClassDiceLoss(self, pred: Optional[Tensor], label: Optional[Tensor], epsilon: float=1e-6):
        nclass = pred.shape[1]
        label = F.one_hot(label.long(), nclass) # convert label to onehot encoding, (N, C, H, W)

        assert pred.shape == label.shape, "Predict & target shape do not match"

        pred = F.softmax(pred, dim=1)
        nchannel = label.shape()[1]
        total_loss = 0

        # traverse the channel to obtain each category's binary classification DiceLoss.
        for i in range(nchannel):
            dice_loss = self.binaryDiceLoss(pred[:, i], label[:, i])
            total_loss += dice_loss
		
		# average dice_loss per class
        return total_loss / nchannel

        


# 测试代码
if __name__ == '__main__':
    pred = torch.as_tensor([
        [0.02, 0.01, 0.01, 0.03],
        [0.04, 0.12, 0.15, 0.07],
        [0.96, 0.93, 0.94, 0.92],
        [0.87, 0.97, 0.96, 0.97],
    ])
    n_pred = torch.stack((pred, pred), dim=0)
    label = torch.as_tensor([
        [0,0,0,0],
        [0,0,0,0],
        [1,1,1,1],
        [1,1,1,1],
    ])
    n_label = torch.stack((label, label), dim=0)
    diceloss = DiceLoss()
    dice_coeff= diceloss.diceCoefficient(n_pred.unsqueeze(dim=1), n_label.unsqueeze(dim=1))
    print(dice_coeff)
