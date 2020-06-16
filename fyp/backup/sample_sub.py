# evaluate using two eyes
import glob
import os
import pandas as pd
import torch
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from pytorch.model_helper import select_model, select_metric
import joblib
import numpy as np
from pytorch.dual_data_helper import create_data_loader, create_dual_test_label_df, create_dual_label_df
from active_learning.extract_features import extract_dual_features


# main_model_dir = "torch_models" #torch_models

main_model_dir = "/media/workstation/Storage/Test/AL/al_val_v01" #torch_models
main_model_dir = "torch_models"
train_name = "dual_xception_higherlr_v03"
model_type = "dual_xception"
best = "best_acc_"
size = 400 
workers = 6

metric_type = "softmax"
model=select_model(model_type, {})

PATH = f"{main_model_dir}/{train_name}/{best}model.pth"
PATH2 = f"{main_model_dir}/{train_name}/{best}metric_fc.pth"

metric_fc = select_metric(metric_type, num_ftr = 1000, num_classes = 5)

model.load_state_dict(torch.load(PATH))
metric_fc.load_state_dict(torch.load(PATH2))
model.eval()
metric_fc.eval()


####

import random
kk = 100
def create_test():
    fdf = create_dual_test_label_df()
    # fdf = fdf.sample(n=, random_state=123)
    data_loader = create_data_loader(fdf, size, batch_size = 2, workers = workers)
    return data_loader,fdf

def create_train():
    df = create_dual_label_df()
    df = df.sample(n=100*kk, random_state=123)
    data_loader = create_data_loader(df, size, batch_size = 2, workers = workers)
    return data_loader, df

test_loader, fdf2 = create_test()
train_loader, fdf = create_train()

f_test, y = extract_dual_features(model, test_loader)
f, y = extract_dual_features(model, train_loader)
fdf["concat_features"] = [j for j in f]
fdf2["concat_features"] = [j for j in f_test]
X_test_final = np.array(fdf2["concat_features"].values.tolist())

print(f[0].shape)
n1 = 80*kk
f1 = fdf.iloc[:n1,:]
f2 = fdf.iloc[n1:, :]

X_train = np.array(f1["concat_features"].values.tolist())
y_train = f1["labels_x"].values
X_test = np.array(f2["concat_features"].values.tolist())
y_test = f2["labels_x"].values
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from evaluate.metrics import accuracy, avg_acc, get_cm
from custom_math.kappa import quadratic_kappa
clf = RandomForestClassifier(max_depth=10, random_state=12)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_test_final = clf.predict(X_test_final)
print(y_test.shape, y_pred_test.shape)
print(y_train.shape, y_pred_train.shape)
print(X_test_final.shape, y_pred_test_final.shape)

y_pred_train, y_train = y_pred_train.astype(int), y_train.astype(int)
y_pred_test, y_test = y_pred_test.astype(int), y_test.astype(int)

acc = accuracy(y_train, y_pred_train)
acc2 = accuracy(y_test, y_pred_test)
qk1 = quadratic_kappa(y_train, y_pred_train)
qk2 = quadratic_kappa(y_test, y_pred_test)
cm1 = get_cm(y_train, y_pred_train)
cm2 = get_cm(y_test, y_pred_test)

print(acc, acc2)
print(qk1, qk2)
print(cm1)
print(cm2)


import joblib
joblib.dump(y_pred_test_final, "y_pred_test_final.pkl")
joblib.dump(X_test_final, "X_test_final.pkl")
fdf2.to_csv("final_test.csv")