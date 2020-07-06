import pandas as pd 
import numpy as np
import time
import os
import glob
import shutil
import random
TRAIN_DIR = "../all_train_300/full_train"
VAL_DIR = "../all_train_300/val"

p = 0.2
os.makedirs(VAL_DIR, exist_ok=True)
for i in range(5):
    img_path = os.path.join(TRAIN_DIR, str(i))
    out_path = os.path.join(VAL_DIR, str(i))
    os.makedirs(out_path, exist_ok=True)
    files = glob.glob(img_path + "/*.jpeg")
    up_to = int(len(files)*p)
    random.seed(i)
    random.shuffle(files)
    for f in files[:up_to]:
        shutil.move(f, out_path)
        

# for label in df1["label"].unique():
#     os.makedirs(f'{TRAIN_DIR}/{label}', exist_ok=True)
 
# print("copying")
# for img, label in zip(df1["image"], df1["label"]):
#     shutil.copy2(f'{INIT_TRAIN_DIR}/{img}.jpeg', f"{TRAIN_DIR}/{label}")
# print("done")
