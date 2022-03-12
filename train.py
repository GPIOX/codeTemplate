'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-10 11:09:27
LastEditors: Cai Weichao
LastEditTime: 2022-03-12 22:24:01
'''

from model.model import Model
import torch
import re
import glob
import yaml
from pathlib import Path
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.dataset import BasicDataset
from utils.txtlogger import TxtLogger

class TrainProcessor:
    def __init__(self, args):
        self.arg = args
        self.increment_path(Path(self.arg.project), mkdir=True)
        self.load_logger()
        self.load_device()
        self.load_model()
        self.load_data()
        self.load_optimizer()
        self.load_loss_func()

    def load_data(self):
        self.data_loader = dict()

        try:
            dataset = BasicDataset()
        except (AssertionError, RuntimeError):
            self.txt_logger.warning('load dataset error!!')

        # Split into train / validation partitions, you can delete when dataset split it before you get
        val_percent = 0.1
        self.n_valid = int(len(dataset) * val_percent)
        self.n_train = len(dataset) - self.n_valid
        train_set, valid_set = random_split(dataset, [self.n_train, self.n_valid])
        # or
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=224)])
        # train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # valid_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

        # set lenth of tqdm
        self.n_train = len(train_set)
        self.n_valid = len(valid_set)

        self.data_loader['train'] = DataLoader(train_set,
                                               batch_size=self.arg.batch_size,
                                               num_workers=self.arg.workers,
                                               shuffle=True)
                                               
        self.data_loader['valid'] = DataLoader(valid_set,
                                               batch_size=self.arg.batch_size,
                                               num_workers=self.arg.workers,
                                               shuffle=False)

    def train(self):
        # save run settings
        with open(f'{self.save_dir}/setting.yaml', 'w') as f:            
            yaml.safe_dump(vars(self.arg), f, sort_keys=False)

        start_epoch = 1

        # load checkpoint
        if self.arg.config_arg['checkpoint']['use']:
            checkpoint_path = self.arg.config_arg['checkpoint']['checkpoint_path']
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint['mdoel'])  # load the learnable parameters of the model 
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # load optimizer parameters 
            start_epoch = checkpoint['epoch']  # set start epoch

        for epoch in range(start_epoch, self.arg.epochs+1):
            self.model.train()
            loss_value = []
            pbar = tqdm(enumerate(self.data_loader['train']), total=len(self.data_loader['train']), ncols=100)

            for _, (images, labels) in pbar:
                images = images.to(self.device)
                labels  = labels.to(self.device)

                # forward
                output = self.model(images)
                loss = self.loss(output, labels)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # statistics
                loss_value.append(loss.data.item())

                # update tqdm info                
                pbar.set_postfix(**{'loss (batch)': loss.item()})  
                pbar.set_description(f'Epoch [{epoch}/{self.arg.epochs}]')

                # save checkpoint
                if epoch % 5 == 0:
                    checkpoint_dict = {
                        'mdoel': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch
                    }

                    torch.save(checkpoint_dict, f'./checkpoint/ckpt_{epoch}.pth')

            mean_loss = np.mean(loss_value)

            # logger and tensorboard
            self.current_epoch = epoch
            self.txt_logger.info(f'Epoch {epoch} train loss : {mean_loss}')
            self.train_logger.add_scalar('train_loss', mean_loss, epoch)
            #self.train_logger.add_scalar('train_other_metric', metric_value, epoch)
            
            #self.adjust_lr(epoch)
            self.evaluate()

    def evaluate(self):
        loss_value = []
        self.model.eval()
        pbar = tqdm(enumerate(self.data_loader['valid']), total=len(self.data_loader['valid']), ncols=100)

        with torch.no_grad():            
            for _, (images, labels) in pbar:
                images = images.to(self.device)
                labels  = labels.to(self.device)

                # forward
                output = self.model(images)
                loss = self.loss(output, labels)

                # statistics 
                loss_value.append(loss.data.item())

                # update tqdm info                
                pbar.set_postfix(**{'loss (batch)': loss.item()})  
                pbar.set_description(f'Valid')
            
            mean_loss = np.mean(loss_value)

            # logger and tensorboard
            self.txt_logger.info(f'Epoch {self.current_epoch} valid loss : {mean_loss}')
            self.valid_logger.add_scalar('valid_loss', mean_loss, self.current_epoch)
            #self.valid_logger.add_scalar('valid_other_metric', metric_value, self.current_epoch)
        

    def show_indicators(self):
        pass

    def adjust_lr(self, epoch):
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def load_logger(self):
        # tensorboard config
        train_logger_path = Path(self.save_dir) / 'train'
        valid_logger_path = Path(self.save_dir) / 'valid'
        train_logger_path.mkdir(parents=True, exist_ok=True)
        valid_logger_path.mkdir(parents=True, exist_ok=True)

        self.train_logger = SummaryWriter(train_logger_path)
        self.valid_logger = SummaryWriter(valid_logger_path)

        # text logger config
        txt_logger = TxtLogger()
        self.txt_logger = txt_logger.load_logger(self.save_dir)      
        

    def load_loss_func(self):
        self.loss = nn.CrossEntropyLoss()

    def load_model(self):
        self.model = Model(**self.arg.config_arg['model_args'])
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        if self.arg.config_arg['optimizer']['select_optim'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), **self.arg.config_arg['optimizer']['optimizer_sgd'])
        elif self.arg.config_arg['optimizer']['select_optim'] == 'Adam':
            self.optimizer = optim.SGD(self.model.parameters(), **self.arg.config_arg['optimizer']['optimizer_adam'])

    def load_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def increment_path(self, path, exist_ok=False, sep='', mkdir=False):
        # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            path = Path(f"{path}{sep}{n}{suffix}")  # increment path
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)  # make directory

        self.save_dir = str(path)
    
