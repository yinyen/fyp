import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch.train_helper import train, validate
from pytorch.param_helper import import_config, create_dir
from pytorch.data_helper import initialize_dataset
from pytorch.model_helper import select_model, select_metric, select_optimizer, select_scheduler
from pytorch.train_helper import PerformanceLog
from pytorch.torch_pipeline import TorchPipeline

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='yaml_config/resnet_v1.yaml',
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
TorchPipeline(**config_kwargs)
