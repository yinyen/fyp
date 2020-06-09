import pandas as pd
import numpy as np
import torch
from pytorch.param_helper import create_dir, create_main_dir, import_config
import glob

class ActiveLearning():
    def __init__(self, **config):
        self.config = config
        self.create_dir(**self.config) #create main dir, return updated_main_dir (increment version if exist)
        full_df, initial_d_unlabel = self.temp_get_filenames(**self.config)

        current_step = 0
        
        ## STEP 0:
        # phase 0: init step j directory and config
        self.create_step_dir(current_step)
        if current_step == 0:
            label_df, val_df, unlabel_df = self.construct_initial_training_set(full_df, initial_d_unlabel, **config)

        # phase 1: Formation of initial cluster


        while current_step < 3:
           
            

            # phase 2:
            
            
            # phase 3: train

            # phase 4: evaluate (using left right)

            # repeat
            current_step += 1

        pass
    
    
    def create_dir(self, main_dir, root_dir, **kwargs):
        updated_main_dir = create_main_dir(f"{root_dir}/{main_dir}")
        self.config["updated_main_dir"] = updated_main_dir

    def temp_get_filenames(self, main_data_dir, **kwargs):
        f1 = glob.glob(f"{main_data_dir}/train/*/*.jpeg")
        f2 = glob.glob(f"{main_data_dir}/val/*/*.jpeg")
        all_files = f1 + f2
        labels = [j.split("/")[-2] for j in all_files]
        full_df = pd.DataFrame(dict(files = all_files, labels = labels))
        initial_d_unlabel = all_files
        return full_df, initial_d_unlabel

    def create_step_dir(self, current_step):
        updated_main_dir = self.config["updated_main_dir"]
        current_step = str(current_step).zfill(3)
        current_step_dir = create_main_dir(f"{updated_main_dir}/step_{current_step}")
        return current_step_dir

    def construct_initial_training_set(self, full_df, d_unlabel, m, n, random_state = 123, **kwargs):
        # d_unlabel: files names of unlabelled training images
        # m is the initial sample size
        # n is the subsequent resampling size
        label_df = full_df.sample(m, random_state = random_state)
        val_df = label_df.copy() # for step 0: validation set = newly added set
        unlabel_df = full_df.drop(label_df.index).copy()
        return label_df, val_df, unlabel_df

    def train(self, model, label_df, val_df):

        pass