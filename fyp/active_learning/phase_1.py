import glob
import pandas as pd
import numpy as np

from preprocessing.load import get_label_from_filename


def phase_1(INITIAL_SAMPLE, TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH, label_df, LIMIT_DEBUG = None):
    img_files = glob.glob(TRAIN_DIR)
    if LIMIT_DEBUG is not None:
        img_files = img_files[:LIMIT_DEBUG]

    unique_label = 0
    training_set = pd.DataFrame()
    while unique_label < 3:
        # 1. Sample 17 images
        x = np.random.choice(img_files, size=INITIAL_SAMPLE, replace = False).tolist()
        img_files = [j for j in img_files if j not in x]
        labels = [get_label_from_filename(f, label_df) for f in x]
        fdf = pd.DataFrame({"img_file": x, "label": labels})
        training_set = pd.concat([training_set, fdf])

        # 2. Calculate number of unique labels
        unique_label = len(training_set.label.unique())
        print("Unique:", unique_label)
        if unique_label < 3:
            print("Repeat sampling!", unique_label)
        
    return training_set, unique_label    