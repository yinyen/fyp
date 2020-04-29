import pandas as pd 
import numpy as np
import time
import os
import glob
import tensorflow as tf

import shutil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Subtract, Concatenate, Dot
from tensorflow.keras.applications.xception import Xception
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# from kappa import quadratic_kappa

def get_label_from_filename(filename, label_df):
    clean_filename = os.path.basename(filename).split(".")[0]
    return label_df.loc[label_df["image"] == clean_filename]["level"].values[0]

label_df = pd.read_csv("trainLabels.csv")

label_df["f"] = [j.split("_")[0] for j in label_df["image"]]
label_df["side"] = [j.split("_")[1] for j in label_df["image"]]

TRAIN_DIR = "all_train"
img_files = glob.glob("Train/Train*/*")

print("labelling")
df1 = pd.DataFrame(dict(img_file = img_files))
df1["label"] = df1["img_file"].apply(lambda x: get_label_from_filename(x, label_df))

print("createdir")
os.makedirs(TRAIN_DIR, exist_ok=True)

for label in df1["label"].unique():
    os.makedirs(f"{TRAIN_DIR}/{label}", exist_ok=True)
 
print("copying")
for img, label in zip(df1["img_file"], df1["label"]):
    shutil.copy2(img, f"{TRAIN_DIR}/{label}")

print("done")



# image_dir = os.path.join('..', 'input')
# df = pd.read_csv(os.path.join(image_dir, 'trainLabels.csv'))
# df['path'] = df['image'].map(lambda x: os.path.join(image_dir,'{}.jpeg'.format(x)))

