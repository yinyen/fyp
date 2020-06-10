import joblib
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image, ImageOps
from PIL import Image

def open_image(img_name, transform = True):
    image2 = Image.open(img_name)
    new_size = image2.size
    max_size = max(image2.size)
    desired_size = max_size
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(image2, padding)
    return new_im


class ImgDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.copy()
        self.transform = transform
        self.targets = df["labels"].astype(int).values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df["files"].values[idx]
        label = self.df["labels"].values[idx]
        image = open_image(img_name)
        if self.transform:
            image = self.transform(image)
        return image, int(label)

def initialize_multi_gen(label_df, val_df, size, batch_size, reweight_sample = -1, reweight_sample_factor = 1, workers = 4):
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

    train_set = ImgDataset(label_df, transform_train)
    val_set = ImgDataset(val_df, transform_val)

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


def create_data_loader(df, size, batch_size = 6, workers = 4):
    transform_val = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
    test_set = ImgDataset(df, transform_val)
    test_gen = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return test_gen