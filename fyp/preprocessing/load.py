import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

def load_label(label_file = "trainLabels.csv"):
    label_df = pd.read_csv("trainLabels.csv")

    label_df["f"] = [j.split("_")[0] for j in label_df["image"]]
    label_df["side"] = [j.split("_")[1] for j in label_df["image"]]
    return label_df

def load_img(img_file, IMG_HEIGHT = 200, IMG_WIDTH = 200):
    img = tf.keras.preprocessing.image.load_img(img_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img / 255.0

def load_img_fast(filenames, worker = 4):
    p = Pool(worker)
    imgs = p.map(load_img, filenames)
    imgs = np.array(imgs)
    return imgs

def get_all_filenames(img_train_dir):
    all_filenames = glob.glob(f"{img_train_dir}/*/*.jpeg")
    return all_filenames

def get_all_test_filenames(img_train_dir):
    all_filenames = glob.glob(f"{img_train_dir}/*.jpeg")
    return all_filenames

def get_label_from_filename(filename, label_df):
    clean_filename = os.path.basename(filename).split(".")[0]
    return label_df.loc[label_df["image"] == clean_filename]["level"].values[0]