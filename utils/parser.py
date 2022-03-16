'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-10 11:25:08
LastEditors: Cai Weichao
LastEditTime: 2022-03-16 22:09:42
'''
import argparse
import yaml
import os

class BaseParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #self.add_other_argument()

    def add_other_argument(self, known=False):
        self.known = False        

        self.parser.add_argument('--seed', type=int, default=42, help='seed')
        self.parser.add_argument('--cfg', type=str, default='./config/config.yaml', help='model config.yaml path')
        self.parser.add_argument('--epochs', default=50, type=int, help='train epochs number')
        self.parser.add_argument('--batch-size', default=12, type=int, help='batch size')
        self.parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
        self.parser.add_argument('--project', default='./runs/exp', help='save to project/name')
        
        _, unparsed  = self.parser.parse_known_args()

        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))

        opt = self.parser.parse_known_args()[0] if self.known else self.parser.parse_args()
        opt.config_arg = self.get_yaml_data(opt.cfg)

        return opt

    def get_yaml_data(self, yaml_file):
        with open(yaml_file) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)

        return config_data
