import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch.param_helper import import_config
from dual.pipeline import DualPipeline

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='yaml_config/new_dual_xception.yaml',
                        help='yaml file path')
    args = parser.parse_args()
    return args

args = parse_args()
config_file = args.config

# load config and initialize config
config_kwargs = import_config(config_file)
print(config_kwargs)

# run entire training
DualPipeline(**config_kwargs)
