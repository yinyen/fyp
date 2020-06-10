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
from active_learning.data_gen import create_data_loader
from active_learning.extract_features import extract_features

train_name = "xception_sgd_test_d500_new_v01"
main_model_dir = "torch_models" #torch_models
model_type = "xception"
best = "best_acc_"

metric_type = "softmax"
model=select_model(model_type, {})

PATH = f"./{main_model_dir}/{train_name}/{best}model.pth"
PATH2 = f"./{main_model_dir}/{train_name}/{best}metric_fc.pth"

metric_fc = select_metric(metric_type, num_ftr = 1000, num_classes = 5)

model.load_state_dict(torch.load(PATH))
metric_fc.load_state_dict(torch.load(PATH2))
model.eval()
metric_fc.eval()


####

main_data_dir = "../all_train_300"
train_dir = "val"

f1 = glob.glob(f"{main_data_dir}/{train_dir}/*/*.jpeg")
all_files = f1
labels = [j.split("/")[-2] for j in all_files]
sides = [os.path.basename(j).split("_")[1].replace("right.jpeg", '1').replace("left.jpeg", '0') for j in all_files]
ids = [os.path.basename(j).split("_")[0] for j in all_files]

df = pd.DataFrame(dict(files = all_files, labels = labels, sides = sides, ids = ids))

df_list = []
for i in df["ids"].unique():
    u1 = df["ids"] == i
    if u1.sum() == 2:
        df_list.append(df.loc[u1])

fdf = pd.concat(df_list)

# fdf = fdf.head(1000)

size = 500 
workers = 6
data_loader = create_data_loader(fdf, size, batch_size = 2, workers = workers)
f, y = extract_features(model, data_loader)
fdf["features"] = [j for j in f]

concat_features = []
for i in range(len(fdf)):
    if i % 2 == 0:
        x = fdf.iloc[i]["features"].tolist()
        x2 = fdf.iloc[i+1]["features"].tolist()
        x3 = x + x2
    else:
        x = fdf.iloc[i]["features"].tolist()
        x2 = fdf.iloc[i-1]["features"].tolist()
        x3 = x + x2
    concat_features.append(x3)

fdf["concat_features"] = concat_features
fdf = fdf.sample(frac=1)

f1 = fdf.iloc[:700, :]
f2 = fdf.iloc[700:, :]

X_train = np.array(f1["concat_features"].values.tolist())
y_train = f1["labels"].values
X_test = np.array(f2["concat_features"].values.tolist())
y_test = f2["labels"].values
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from evaluate.metrics import accuracy, avg_acc, get_cm
from custom_math.kappa import quadratic_kappa
# X, y = make_classification(n_samples=1000, n_features=4,
#                             n_informative=2, n_redundant=0,
#                             random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=10, random_state=12)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(y_pred_train.shape, y_pred_test.shape)

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
