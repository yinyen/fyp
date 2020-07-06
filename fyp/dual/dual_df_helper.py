import os
import glob
import pandas as pd
import numpy as np

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

    df2 = df.copy()
    df2.columns = ['ids', 'files_y', 'labels_y', 'sides_y', 'files_x', 'labels_x', 'sides_x']
    df3 = pd.concat([df,df2])
    df3["labels_x"] = df3["labels_x"].astype(int)
    df3["sides_x"] = df3["sides_x"].astype(int)
    df3["labels_y"] = df3["labels_y"].astype(int)
    df3["sides_y"] = df3["sides_y"].astype(int)
    return df3


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

    df2 = df.copy()
    df2.columns = ['ids', 'files_y', 'labels_y', 'sides_y', 'files_x', 'labels_x', 'sides_x']
    df3 = pd.concat([df,df2])
    df3["labels_x"] = df3["labels_x"].astype(int)
    df3["sides_x"] = df3["sides_x"].astype(int)
    df3["labels_y"] = df3["labels_y"].astype(int)
    df3["sides_y"] = df3["sides_y"].astype(int)
    return df


def split_dual_df(dual_df, p = 0.2, seed = 123):
    d2 = dual_df.sample(frac = p, random_state = seed)
    d1 = dual_df.drop(d2.index)
    return d1, d2 # train, val
