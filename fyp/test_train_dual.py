import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch.param_helper import import_config
from dual.pipeline import DualPipeline

import argparse
from active_learning.new_pipeline import ActiveDualPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='yaml_config/train_resnext.yaml',
                        help='yaml file path')
    args = parser.parse_args()
    return args

args = parse_args()
config_file = args.config

# load config and initialize config
config_kwargs = import_config(config_file)
print(config_kwargs)

import pandas as pd
from dual.dual_df_helper import create_dual_label_df

dual_df = create_dual_label_df("..", ["new_all_train_400"])
print(dual_df.shape)
n2 = dual_df.shape[0]//2-1
# print(.files_x.unique().shape)
# print(dual_df.files_x.unique().shape)
# raise Exception()
label_df = pd.read_csv("torch_DualAL/dual_al_ui_v00/step_024/label_df.csv")
val_df = pd.read_csv("torch_DualAL/dual_al_ui_v00/step_024/val_df.csv")
unlabel_df = pd.read_csv("torch_DualAL/dual_al_ui_v00/step_024/unlabel_df.csv")

# adf = unlabel_df.sample(n=2000, random_state = 123)
adf = dual_df.iloc[:n2, :].sample(n=3000, random_state = 123)
label_df = adf.sample(n=1500, random_state = 123)
val_df = adf.drop(label_df.index)
print(label_df.shape)
print(val_df.shape)
# run entire training
ActiveDualPipeline(d_train = label_df, d_val = val_df, model = None, **config_kwargs)
