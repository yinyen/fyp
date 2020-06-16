# evaluate using two eyes
import os, glob
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import torch
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch.dual_data_helper import create_dual_label_df, create_dual_test_label_df, create_data_loader, split_dual_df
from pytorch.model_helper import select_model, select_metric
from active_learning.extract_features import extract_features
import time

def extract_dual_features(model, data_loader):
    feature_list = []
    y_list = []
    img_list = []
    for i, (img1, img2, input1, input2, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        input1, input2, target = input1.cuda(), input2.cuda(), target.cuda()
        output = model(input1, input2) 
        x, y = output.cpu().detach().numpy().tolist(), target.cpu().detach().numpy().tolist()
        img1 = list(img1)
        feature_list += x
        y_list += y
        img_list += img1

    N = len(y_list)
    k = N // 2

    double = 1
    if double:
        new_feature_list1 = [feature_list[i] + feature_list[k+i] for i in range(k)]
        new_feature_list2 = [feature_list[k+i] + feature_list[i] for i in range(k)]
        new_feature_list = new_feature_list1 + new_feature_list2
        print(len(new_feature_list), k, len(new_feature_list[0]))
        features = np.array(new_feature_list) #.reshape(-1, 8192)
    else:
        features = np.array(feature_list) #.reshape(-1, 8192)
    X = features.reshape((features.shape[0], -1))

    y = np.array(y_list)
    return X, y, img_list

t0 = time.time()

main_model_dir = "/media/workstation/Storage/Test/Dual_AL/dual_al_test_v18" #
train_name = "step_020"
metric_type = "softmax"
model_type = "dual_xception"
best = "best_acc_"
size = 400 
workers = 6

PATH = f"{main_model_dir}/{train_name}/{best}model.pth"
PATH2 = f"{main_model_dir}/{train_name}/{best}metric_fc.pth"

model = select_model(model_type, {})
metric_fc = select_metric(metric_type, num_ftr = 1000, num_classes = 5)

model.load_state_dict(torch.load(PATH))
metric_fc.load_state_dict(torch.load(PATH2))
model.eval()
metric_fc.eval()

train_dual_df = create_dual_label_df(main_data_dir = "../all_train_300", train_dir_list = ["full_train", "val"])
train_dual_df, val_dual_df = split_dual_df(train_dual_df, p = 0.2, seed = 123)
test_dual_df = create_dual_test_label_df(main_data_dir = "/media/workstation/Storage/Test/fp/test_300")

rs = 123
# train_dual_df = train_dual_df.sample(50, random_state = rs)
# val_dual_df = val_dual_df.sample(30, random_state = rs)
# test_dual_df = test_dual_df.sample(50, random_state = rs)

print(train_dual_df.shape)
print(train_dual_df["labels_x"].value_counts())

####
train_data_loader = create_data_loader(train_dual_df, size, batch_size = 2, workers = workers, return_5 = True)
val_data_loader = create_data_loader(val_dual_df, size, batch_size = 2, workers = workers, return_5 = True)
test_data_loader = create_data_loader(test_dual_df, size, batch_size = 2, workers = workers, return_5 = True)

X_train, y_train, img_train = extract_dual_features(model, train_data_loader)
X_val, y_val, img_val = extract_dual_features(model, val_data_loader)
X_test, y_test, img_test = extract_dual_features(model, test_data_loader)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# raise Exception()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from evaluate.metrics import accuracy, avg_acc, get_cm
from custom_math.kappa import quadratic_kappa
# clf = RandomForestClassifier(max_depth=10, random_state=12)


def evaluate(X_train, y_train, X_test, y_test):
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    y_pred_train, y_train = y_pred_train.astype(int), y_train.astype(int)
    y_pred_test, y_test = y_pred_test.astype(int), y_test.astype(int)

    acc = accuracy(y_train, y_pred_train)
    acc2 = accuracy(y_test, y_pred_test)
    avg_acc1 = avg_acc(y_train, y_pred_train)
    avg_acc2 = avg_acc(y_test, y_pred_test)
    qk1 = quadratic_kappa(y_train, y_pred_train)
    qk2 = quadratic_kappa(y_test, y_pred_test)
    cm1 = get_cm(y_train, y_pred_train)
    cm2 = get_cm(y_test, y_pred_test)

    print("Accuracy:", acc, acc2)
    print("AvgAcc:", avg_acc1, avg_acc2)
    print("QK:", qk1, qk2)
    print(cm1)
    print(cm2)

def get_test_result(X_test, img_test):
    img_name = [os.path.basename(j).split(".")[0] for j in img_test]
    y_pred = clf.predict(X_test)
    df = pd.DataFrame(dict(image = img_name, level = y_pred))
    df = df.sort_values("image")
    return df

# clf = LogisticRegression()
clf = RandomForestClassifier(max_depth=10, random_state=12)
clf.fit(X_train, y_train)

out_dir = "dual_features"
packet = [X_train, y_train, X_val, y_val, X_test, y_test]
joblib.dump(packet, f"{out_dir}/packet_dual.pkl")

evaluate(X_train, y_train, X_val, y_val)

sub_df = get_test_result(X_test, img_test)
sub_df.to_csv("submission_test_rf.csv", index = False)
print(sub_df.shape)
#53576
t1 = time.time()
print("Time taken:", t1-t0)