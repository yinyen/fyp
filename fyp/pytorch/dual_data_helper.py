import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random

import joblib
import os, glob
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image, ImageOps
from PIL import Image
from skimage import io, transform
from preprocessing.image_preprocess import preprocess_eye

def create_dual_label_df(main_data_dir = "../all_train_300", train_dir_list = ["full_train", "val"]):
    all_files = []
    for train_dir in train_dir_list:
        f1 = glob.glob(f"{main_data_dir}/{train_dir}/*/*.jpeg")
        all_files += f1

    labels = [j.split("/")[-2] for j in all_files]
    sides = [os.path.basename(j).split("_")[1].replace("right.jpeg", '1').replace("left.jpeg", '0') for j in all_files]
    sides = [int(j) for j in sides]
    ids = [os.path.basename(j).split("_")[0] for j in all_files]

    df = pd.DataFrame(dict(files = all_files, labels = labels, sides = sides, ids = ids))
    left_df = df.loc[df["sides"]==0].sort_values("ids")
    right_df = df.loc[df["sides"]==1].sort_values("ids")
    df = pd.merge(left_df, right_df, left_on = "ids", right_on = "ids", how = "outer")
    df["files_y"] = df["files_y"].fillna(df["files_x"])
    df["labels_y"] = df["labels_y"].fillna(df["labels_x"])
    df["sides_y"] = df["sides_y"].fillna(df["sides_x"])
    df["files_x"] = df["files_x"].fillna(df["files_y"])
    df["labels_x"] = df["labels_x"].fillna(df["labels_y"])
    df["sides_x"] = df["sides_x"].fillna(df["sides_y"])
    df = df.set_index("ids").reset_index()
    return df

def create_dual_test_label_df(main_data_dir = "/media/workstation/Storage/Test/fp/test_300"):
    all_files = glob.glob(f"{main_data_dir}/*.jpeg")

    sides = [os.path.basename(j).split("_")[1].replace("right.jpeg", '1').replace("left.jpeg", '0') for j in all_files]
    sides = [int(j) for j in sides]
    ids = [os.path.basename(j).split("_")[0] for j in all_files]

    df = pd.DataFrame(dict(files = all_files, sides = sides, ids = ids))
    df['labels'] = 0
    
    left_df = df.loc[df["sides"]==0].sort_values("ids")
    right_df = df.loc[df["sides"]==1].sort_values("ids")
    df = pd.merge(left_df, right_df, left_on = "ids", right_on = "ids", how = "outer")
    df["files_y"] = df["files_y"].fillna(df["files_x"])
    df["labels_y"] = df["labels_y"].fillna(df["labels_x"])
    df["sides_y"] = df["sides_y"].fillna(df["sides_x"])
    df["files_x"] = df["files_x"].fillna(df["files_y"])
    df["labels_x"] = df["labels_x"].fillna(df["labels_y"])
    df["sides_x"] = df["sides_x"].fillna(df["sides_y"])
    df = df.set_index("ids").reset_index()
    return df

def split_dual_df(dual_df, p = 0.2, seed = 123):
    d2 = dual_df.sample(frac = p, random_state = seed)
    d1 = dual_df.drop(d2.index)
    return d1, d2 # train, val

def open_image(img_name, transform_1 = False, transform_2 = True):
    image2 = Image.open(img_name)
    try:
        if transform_2:
            new_size = image2.size
            max_size = max(image2.size)
            desired_size = max_size
            delta_w = desired_size - new_size[0]
            delta_h = desired_size - new_size[1]
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            new_im = ImageOps.expand(image2, padding)
    except:
        new_im = image2
    return new_im

class DualImgDataset(Dataset):
    def __init__(self, df, transform=None, apply_raw_transform = False, return_5 = False):
        self.df = df.copy()
        self.transform = transform
        self.apply_raw_transform = apply_raw_transform
        self.return_5 = return_5
        self.targets = np.array(df["labels_x"].astype(int).values.tolist() + df["labels_y"].astype(int).values.tolist())
        self.n = self.df.shape[0]

    def __len__(self):
        return self.df.shape[0]*2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx // self.n == 0:
            idx = idx % self.n
            img_name_1 = self.df["files_x"].values[idx]
            img_name_2 = self.df["files_y"].values[idx]
            label = self.df["labels_x"].values[idx]
            image_1 = open_image(img_name_1)
            image_2 = open_image(img_name_2)
            if self.transform:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
        elif idx // self.n == 1:
            idx = idx % self.n
            img_name_1 = self.df["files_y"].values[idx]
            img_name_2 = self.df["files_x"].values[idx]
            label = self.df["labels_y"].values[idx]
            image_1 = open_image(img_name_1)
            image_2 = open_image(img_name_2)

            # if self.apply_raw_transform:
            #     pass

            if self.transform:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
        
        if self.return_5:
            return img_name_1, img_name_2, image_1, image_2, int(label)
        else:
            return image_1, image_2, int(label)


def initialize_dual_gen(train_label_df, val_label_df, size, batch_size, reweight_sample = -1, reweight_sample_factor = 1, workers = 4):
    transform_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    transform_val = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    train_set = DualImgDataset(train_label_df, transform_train)
    val_set = DualImgDataset(val_label_df, transform_val)

    if reweight_sample != -1:
        targets = np.array(train_set.targets)
        total_n = targets.shape[0]
        resample_total = int(total_n*reweight_sample_factor)
        s_targets = targets
        samples_weight = np.array([1/np.mean(s_targets == i) for i in np.unique(s_targets)])
        samples_weight = np.array([samples_weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, resample_total, replacement = True)
        train_gen = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers)
    else:
        train_gen = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_gen = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_gen, val_gen


def create_data_loader(df, size, batch_size = 6, workers = 4, return_5 = False):
    transform_val = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
    test_set = DualImgDataset(df, transform_val, return_5 = return_5)
    test_gen = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return test_gen
