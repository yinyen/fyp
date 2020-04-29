import pandas as pd 
import numpy as np
import time
import os
import glob
import shutil
from preprocessing.load import get_label_from_filename

label_df = pd.read_csv("trainLabels.csv")
label_df["f"] = [j.split("_")[0] for j in label_df["image"]]
label_df["side"] = [j.split("_")[1] for j in label_df["image"]]
label_df["label"] = label_df["level"].apply(lambda x: str(x))
df1 = label_df.copy()

INIT_TRAIN_DIR = "../train"
TRAIN_DIR = "../all_train"
os.makedirs(TRAIN_DIR, exist_ok=True)
for label in df1["label"].unique():
    os.makedirs(f'{TRAIN_DIR}/{label}', exist_ok=True)
 
print("copying")
for img, label in zip(df1["image"], df1["label"]):
    shutil.copy2(f'{INIT_TRAIN_DIR}/{img}.jpeg', f"{TRAIN_DIR}/{label}")
print("done")

