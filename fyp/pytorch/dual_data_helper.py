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
from skimage import io, transform
from preprocessing.image_preprocess import load_transform_image


class DualImgDataset(Dataset):
    def __init__(self, df, size, transform=None, return_5 = -1):
        self.df = df.copy()
        self.transform = transform
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
            image_1 = load_transform_image(img_name_1, size)
            image_2 = load_transform_image(img_name_2, size)
            if self.transform:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
        elif idx // self.n == 1:
            idx = idx % self.n
            img_name_1 = self.df["files_y"].values[idx]
            img_name_2 = self.df["files_x"].values[idx]
            label = self.df["labels_y"].values[idx]
            image_1 = load_transform_image(img_name_1, size)
            image_2 = load_transform_image(img_name_2, size)

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
            transforms.RandomAffine(
                degrees=(-180,180),
                scale=(0.8889, 1.0),
                shear=(-36,36)
            ),
            # transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(contrast=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    transform_val = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
