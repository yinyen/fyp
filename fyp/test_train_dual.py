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

from tqdm import tqdm
import numpy as np
import pandas as pd
from dual.dual_df_helper import create_dual_label_df
from pytorch.model_helper import select_model
from pytorch.param_helper import import_config
from dual.training import convert_pred
from dual.dual_gen_helper import initialize_dual_gen
# from keras.utils import to_categorical
path0 = "/home/ubuntu/fyp/fyp/torch_DualAL/dual_al_random_v01/"
path = "/home/ubuntu/fyp/fyp/torch_DualAL/dual_al_random_v01/step_019/"
path1 = os.path.join(path0, "active_learning_config.yaml")
config = import_config(path1)
model_type = config.get("model_config").get("model_type") 
model_kwargs = config.get("model_config").get("model_kwargs")
previous_model_path = os.path.join(path, "new_single_resnext50_imagenet_400", "best_qk_model.pth")
print(model_type)
model = select_model(model_type, model_kwargs)
model.load_state_dict(torch.load(previous_model_path))

label_df = pd.read_csv(os.path.join(path, "label_df.csv"))
val_df = pd.read_csv(os.path.join(path, "val_df.csv"))
print(label_df.shape, val_df.shape)
train_gen, val_gen = initialize_dual_gen(label_df, val_df, size = 400, batch_size = 2, reweight_sample = 0, reweight_sample_factor = 1, workers = 4, single_mode = 0, load_only = 1)

def indices_to_one_hot(data, nb_classes = 5):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def extract_xy(model, data_gen, one_hot = False):
    features_list = []
    y_pred = []
    y_true = []
    features_extractor = nn.Sequential(*list(model.children())[:-1])
    for i, (input1, input2, target) in tqdm(enumerate(data_gen), total=len(data_gen)):
        input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
        output = features_extractor(input1) 
        features_list.append(np.array(output.tolist()))
        target_to_add = target.int().tolist()
        y_true += target_to_add
    features = np.array(features_list)
    features = features.reshape((features.shape[0]*features.shape[1], -1))
    y_true = np.array(y_true)
    if one_hot:
        y_true = indices_to_one_hot(y_true)
    return features, y_true

X, y = extract_xy(model, train_gen)
X_val, y_val = extract_xy(model, val_gen)


t0 = time.time()


clf = LogisticRegression(random_state=0, max_iter=200).fit(X, y)
clf.fit(X, y)

y_pred = clf.predict(X_val)
ypb = clf.predict_proba(X_val)
print(ypb)
print(ypb.shape)
from evaluate.kappa import quadratic_kappa
qk = quadratic_kappa(y_val, y_pred)   
print(qk)
print(time.time()-t0)
# df = pd.DataFrame(dict(y1 = y, y2 = yp))
# print(df)

# dual_df = create_dual_label_df("..", ["new_all_train_400"])
# print(dual_df.shape)
# n2 = dual_df.shape[0]//2-1
# print(.files_x.unique().shape)
# print(dual_df.files_x.unique().shape)
# raise Exception()
# label_df = pd.read_csv("torch_DualAL/dual_al_ui_v00/step_024/label_df.csv")
# val_df = pd.read_csv("torch_DualAL/dual_al_ui_v00/step_024/val_df.csv")
# unlabel_df = pd.read_csv("torch_DualAL/dual_al_ui_v00/step_024/unlabel_df.csv")

# adf = unlabel_df.sample(n=2000, random_state = 123)
# adf = dual_df.iloc[:n2, :].sample(n=3000, random_state = 123)
# label_df = adf.sample(n=1500, random_state = 123)
# val_df = adf.drop(label_df.index)
# print(label_df.shape)
# print(val_df.shape)
# # run entire training
# ActiveDualPipeline(d_train = label_df, d_val = val_df, model = None, **config_kwargs)
