import torch
import torch.nn.functional as F
from torch import nn

from pytorch.model_helper import select_model
from dual.dual_df_helper import *
from dual.dual_gen_helper import *
from tqdm import tqdm

import time

import numpy as np
import os 
import shutil

shutil.move("testtwo", "test1")
raise Excetion()
# z = np.random.rand(30000, 2048)

# print(z.shape)

# import sys

# y = sys.getsizeof(z)
# print(y)

# print("sleep")
# time.sleep(10)


model_kwargs = { "num_ftrs": 1000, "num_classes": 1 }
premodel_path = "torch_DualAL/dual_al_ui_v17/step_000/new_single_resnext50_imagenet_400/best_qk_model.pth"

main_data_dir = ".."
train_dir_list = ["new_all_train_400"]

load_only = 1
dual_df = create_dual_label_df(main_data_dir = main_data_dir, train_dir_list = train_dir_list)
d1, d_train = split_dual_df(dual_df, p = 0.002, seed = 321) # use 20k*0.05 = 1k samples for training
d1, d_val = split_dual_df(dual_df, p = 0.002, seed = 321) # use 20k*0.05 = 1k samples for training
train_gen, val_gen = initialize_dual_gen(d_train, d_val, 400, 4, 0, 1, 0, single_mode = 0, load_only = load_only) # force double image for now
    
model = select_model(model_type = "single_resnext101", model_kwargs = model_kwargs)
model.load_state_dict(torch.load(premodel_path))

features_list = []
features_extractor = nn.Sequential(*list(model.children())[:-1])
for i, (input1, input2, target) in tqdm(enumerate(train_gen), total=len(train_gen)):
    input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
    output = features_extractor(input1) 
    features_list.append(np.array(output.tolist()))

features = np.array(features_list)
features = features.reshape((features.shape[0]*features.shape[1], -1))


# m = torch.nn.Softmax(dim=1)
# input = torch.randn(2, 3).cuda()
# print(input)

# d = asm(input)
# print("++++++++++++++++++")
# print(d)

# output = F.log_softmax(input)
# print(output)

# m = nn.MaxPool2d(3, stride=2)
# # pool of non-square window
# # m = nn.MaxPool2d((3, 2), stride=(2, 2))
# input = torch.randn(20, 16, 50, 32)
# output = m(input)
# print(output.shape)

# import torch
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)

# model.fc = nn.Linear(2048, 5)
# print(model)