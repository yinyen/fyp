import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch.param_helper import import_config
from pytorch.dual_torch_pipeline import DualTorchPipeline

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='yaml_config/dual_xception.yaml',
                        help='yaml file path')
    args = parser.parse_args()
    return args

args = parse_args()
config_file = args.config
# config_file = "yaml_config/resnet_v1.yaml"

# load config and initialize config
config_kwargs = import_config(config_file)
print(config_kwargs)

# run entire training
DualTorchPipeline(**config_kwargs)
