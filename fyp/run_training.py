import os
import glob
import numpy as np
import pandas as pd
from active_learning.phase import do_phase_1, do_phase_2

################
## Configuration
#################
TRAIN_DIR = "../all_train/*/*.jpeg"
IMG_WIDTH = 200
IMG_HEIGHT = 200
NUM_CLASS = 5
INITIAL_SAMPLE = 17 #6.9%
EVERY_SAMPLE = 20

PHASE_2 = "Phase_2_Output_v1_test01"
ITERATION = 1
BATCH_SIZE = 32
EPOCH = 6
lr = 0.0001


################
## PHASE 1
################
training_set, label_df, all_filenames = do_phase_1(TRAIN_DIR, INITIAL_SAMPLE, IMG_HEIGHT, IMG_WIDTH)

################
## PHASE 2
################
new_training_set = do_phase_2(PHASE_2, ITERATION, training_set, label_df, EVERY_SAMPLE,
                    NUM_CLASS, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, lr, EPOCH)