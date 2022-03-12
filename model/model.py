'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-10 15:30:18
LastEditors: Cai Weichao
LastEditTime: 2022-03-11 20:09:54
'''
import torch
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_channel=3, num_class=4):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_channel, 6)

    def forward(self, x):
        pass








