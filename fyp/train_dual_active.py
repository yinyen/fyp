import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch.param_helper import import_config, create_dir
from active_learning.dual_pipeline import DualActiveLearning
from active_learning.dual_random_pipeline import DualRandomActiveLearning
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='yaml_config/al_dual_xception.yaml',
                        help='yaml file path')
    args = parser.parse_args()
    return args

args = parse_args()
config_file = args.config

# load config and initialize config
config_kwargs = import_config(config_file)
print(config_kwargs)

# run entire training
if config_kwargs["style"] == "yy":
    print("Using custom active learning: yy")
    DualActiveLearning(**config_kwargs)
elif config_kwargs["style"] == "random":
    print("Using custom active learning: Random")
    DualRandomActiveLearning(**config_kwargs)
