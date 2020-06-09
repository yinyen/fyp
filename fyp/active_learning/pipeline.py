import pandas as pd
import numpy as np
import torch
from pytorch.param_helper import create_dir, create_main_dir, import_config
from active_learning.model_pipeline import TrainPipeline
from active_learning.data_gen import create_data_loader
from active_learning.extract_features import extract_features
import glob

class ActiveLearning():
    def __init__(self, **config):
        self.config = config
        self.create_dir(**self.config) #create main dir, return updated_main_dir (increment version if exist)
        full_df, initial_d_unlabel = self.temp_get_filenames(**self.config)

        current_step = 0
        
        ## STEP 0:
        # phase 0: init step j directory and config
        current_step_dir = self.create_step_dir(current_step)
        if current_step == 0:
            label_df, val_df, unlabel_df = self.construct_initial_training_set(full_df, initial_d_unlabel, **config)

        # phase 1: Formation of initial cluster
        model, metric_fc = self.train(current_step_dir, label_df, val_df, None, **self.config)
        centroid = self.extract_features_and_form_initial_clusters(model, label_df, **config)

        print(centroid)
        raise Exception()

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
        unlabel_df = full_df.drop(label_df.index).copy()
        while len(label_df["labels"].unique()) < 5:
            sub_df = unlabel_df.sample(n, random_state = random_state)
            label_df = label_df.append(sub_df)
            unlabel_df = unlabel_df.drop(sub_df.index).copy()
        val_df = unlabel_df.sample(n, random_state = random_state) # val_df is for next step training, but used for this step validation
        unlabel_df = unlabel_df.drop(val_df.index).copy()
        return label_df, val_df, unlabel_df

    def extract_features_and_form_initial_clusters(self, model, label_df, size, workers, **kwargs):
        data_loader = create_data_loader(label_df, size, batch_size = 6, workers = workers)
        f, y = extract_features(model, data_loader)
        label_df["features"] = [j for j in f]

        centroid = {}
        for i in range(5):
            uf = label_df["labels"] == str(i)
            cent = label_df.loc[uf, "features"].values.mean()
            centroid[i] = cent
        return centroid


    def train(self, current_step_dir, label_df, val_df, model, **kwargs):
        CNN = TrainPipeline(step_dir = current_step_dir, label_df = label_df, val_df = val_df, model = None, **kwargs)
        model, metric_fc = CNN.get_model()
        return model, metric_fc