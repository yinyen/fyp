import os
import argparse
from pytorch.param_helper import import_config
from active_learning.new_pipeline import ActiveLearning

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='yaml_config/al_demo.yaml',
                        help='yaml file path')
    args = parser.parse_args()
    return args

args = parse_args()
config_file = args.config

# load config and initialize config
config_kwargs = import_config(config_file)
print(config_kwargs)

# run entire training
ActiveLearning(**config_kwargs)
