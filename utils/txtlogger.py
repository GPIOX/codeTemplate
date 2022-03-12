'''
Descripttion: 
version: 
Author: Cai Weichao
Date: 2022-03-12 18:21:58
LastEditors: Cai Weichao
LastEditTime: 2022-03-12 20:14:16
'''
import logging
from pathlib import Path

class TxtLogger:
    def __init__(self):
        pass

    def load_logger(self, save_dir):
        # set logger output string
        format = '%(asctime)s - %(levelname)s - %(message)s'
        datefmt='%Y-%m-%d %H:%M:%S'

        logging.basicConfig(level=logging.INFO, format=format, datefmt=datefmt)
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(Path(save_dir) / "textlog.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(format, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
        
