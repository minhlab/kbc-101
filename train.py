'''
Created on Nov 17, 2014

@author: Minh Ngoc Le
'''
import os
from pylearn2.config import yaml_parse


if __name__ == '__main__':
    home_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(home_dir, "train.yaml")
    with open(train_path, "rt") as f:
        train_str = f.read()
    train = yaml_parse.load(train_str)
    train.main_loop()