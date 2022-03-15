'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-10 15:42:42
LastEditors: Cai Weichao
LastEditTime: 2022-03-15 20:57:22
'''
import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self):
        super(BasicDataset, self).__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

'''
name: get_kfold_data
brief: Used to obtain the data required for k-fold cross-validation,
       The premise is that all data has been converted to tensor 
author: Cai Weichao
param {*} k_fold
param {*} ith_fold
param {*} images
param {*} labels
return {*}
'''
def tensor_get_kfold_data(k_fold, ith_fold, images, labels):
    fold_size = images.shape[0] // k_fold  # Sample size included in each fold 
    
    val_start = ith_fold * fold_size
    if ith_fold != k_fold - 1:
        val_end = (ith_fold + 1) * fold_size
        images_valid, labels_valid = images[val_start:val_end], labels[val_start:val_end]
        images_train = torch.cat((images[0:val_start], images[val_end:]), dim = 0)
        labels_train = torch.cat((labels[0:val_start], labels[val_end:]), dim = 0)
    else:  
        images_valid, labels_valid = images[val_start:], labels[val_start:] # If it is not divisible, put the multiple cases in the last fold 
        images_train = images[0:val_start]
        labels_train = labels[0:val_start]
        
    return images_train, labels_train, images_valid, labels_valid





