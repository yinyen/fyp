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



label_df = pd.read_csv("trainLabels.csv")

label_df["f"] = [j.split("_")[0] for j in label_df["image"]]
label_df["side"] = [j.split("_")[1] for j in label_df["image"]]

g = label_df.groupby("f")["level"].count()
y = label_df.groupby("level").count()
print(y)

def load_img(img_file, IMG_HEIGHT = 200, IMG_WIDTH = 200):
    img = tf.keras.preprocessing.image.load_img(img_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img / 255.0

def euclidean_distance(x, y):
    dist = np.linalg.norm(x-y)
    return dist

def get_label_from_filename(filename, label_df):
    clean_filename = os.path.basename(filename).split(".")[0]
    return label_df.loc[label_df["image"] == clean_filename]["level"].values[0]

BATCH_SIZE = 128

################
## PHASE 1
################
# data_dir = "Train"
INITIAL_SAMPLE = 17 #6.9%
SELECT_N = 10
img_files = glob.glob("Train/Train*/*")
img_files = img_files[:100]

unique_label = 0
training_set = pd.DataFrame()
while unique_label < 3:
    # 1. Sample 17 images
    x = np.random.choice(img_files, size=INITIAL_SAMPLE, replace = False)
    loaded_imgs = [load_img(j) for j in x]
    labels = [get_label_from_filename(f, label_df) for f in x]
    fdf = pd.DataFrame({"img_file": x, "label": labels})
    training_set = pd.concat([training_set, fdf])

    # 2. Calculate number of unique labels
    unique_label = len(fdf.label.unique())
    print("Unique:", unique_label)
    if unique_label < 3:
        print("Repeat sampling!", unique_label)
    
print(training_set.shape)
print(training_set.label.unique())

# # 2. Compute euclidean distance between first sampled image and all other "unlabelled" images
# dist_list = []
# for img_file in img_files:
#     x = load_img(img_file)
#     d = euclidean_distance(first_img, x)
#     dist = {}    
#     dist["img_file"] = img_file
#     dist["distance"] = d
#     dist_list.append(dist)


# fdf = pd.DataFrame(dist_list)
# print(fdf.head())

# # 3. sort by distance, and remove top 2% after ranking
# fdf = fdf.sort_values("distance", ascending = False)
# idx = int(fdf.shape[0]*0.02)
# fdf = fdf.iloc[idx:,]
# print(fdf.head())

# # 4. Select top N
# selected_df = fdf.head(SELECT_N)

# # 5. LABEL the N images
# selected_df["label"] = selected_df["img_file"].apply(lambda x: get_label_from_filename(x, label_df))
# print(selected_df.sort_values("label", ascending = False).head(10))
# print(selected_df.shape)

# training_set = selected_df.copy()
################
## PHASE 2
################
## DELETE TO RETRY
try:
    shutil.rmtree("Selected_Train")
except:
    print("Selected_Train does not exist.")

##### LOOP
# Train a model using the labelled images
os.makedirs("Selected_Train", exist_ok=True)

for label in training_set["label"].unique():
    os.makedirs(f"Selected_Train/{label}", exist_ok=True)

for img, label in zip(training_set["img_file"], training_set["label"]):
    shutil.copy2(img, f"Selected_Train/{label}")