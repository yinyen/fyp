import glob
import json
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from pytorch.param_helper import create_dir, create_main_dir, import_config
from dual.dual_df_helper import create_dual_label_df, split_dual_df
from dual.dual_gen_helper import initialize_dual_gen, create_data_loader
from dual.pipeline import DualPipeline
from active_learning.extract_features import extract_features, predict
from active_learning.metrics import unfamiliarity_index
from evaluate.metrics import accuracy, avg_acc, get_cm
from evaluate.kappa import quadratic_kappa


class ActiveDualPipeline(DualPipeline):
    def __init__(self, d_train, d_val, model, **kwargs):
        cudnn.benchmark = True
        
        train_name = kwargs.get("train_name")
        main_model_dir = kwargs.get("main_model_dir")
        train_name_updated = create_dir(train_name, main_model_dir)
        self.model_path = f"{main_model_dir}/{train_name_updated}"
        self.log_path = f"{main_model_dir}/{train_name_updated}/log.csv"

        print("=====================================")
        print("SAVING MODEL TO >>>", self.model_path)
        print("=====================================")

        train_loader, val_loader = self.init_dataset(d_train, d_val, **kwargs)
        model, optimizer, scheduler = self.init_model(model, **kwargs)
        criterion = self.init_loss(**kwargs)

        if kwargs.get("retrain"):
            self.init_retrain(model, **kwargs)

        self.dump_config(self.model_path, kwargs)
        self.train(train_loader, val_loader, model, criterion, optimizer, scheduler, **kwargs)

    def init_dataset(self, d_train, d_val, batch_size, size, workers, reweight_sample = 1, reweight_sample_factor = 2, single_mode = 0, load_only = 0, **kwargs):
        # initialize dataset generators
        print("Train:", d_train.shape, "Val:", d_val.shape)
        train_gen, val_gen = initialize_dual_gen(d_train, d_val, size, batch_size, reweight_sample, reweight_sample_factor, workers, single_mode = 0, load_only = load_only) # force double image for now
        return train_gen, val_gen


class ActiveLearning():
    def __init__(self, **config):
        self.result_df = None
        self.result_df2 = None

        self.al_config = config.get("al_config")
        self.model_config = config.get("model_config")
        self.max_steps = self.al_config.get("max_steps")

        # A0: Create Active Learning directory
        self.create_dir(**self.al_config) # create main dir, return updated_main_dir (increment version if exist)

        # A1: Partition 
        dual_df = create_dual_label_df(main_data_dir = self.model_config.get("main_data_dir"), train_dir_list = self.model_config.get("train_dir_list"))
        full_df, val_df = split_dual_df(dual_df, p = None, seed = 321, n = self.al_config.get("val_n_sample")) 
        full_df["ui"] = 0
        full_df["features"] = 0

        # A2: Active Learning
        # full_df, current_step, centroid, unlabel_df, model, val_df, previous_model_path
        if self.al_config.get("style") == "ui":
            self.ui_active_learning(full_df, val_df)
        elif self.al_config.get("style") == "random":
            self.random_active_learning(full_df, val_df, self.al_config.get("random_state"))

    
    def ui_active_learning(self, full_df, val_df):
        model = None
        current_step = 0
        while current_step < self.max_steps:

            # phase 0: init step 0 directory 
            current_step_dir = self.create_step_dir(current_step)

            # phase 1: construct new label_df, unlabel_df, to_add_df 
            if current_step == 0:
                # Initialization and form df
                label_df, unlabel_df = self.construct_initial_training_set(full_df, **self.al_config)
                to_add_df = label_df.copy()
            else: 
                # Compute UI and form df
                # compute unfamiliarity index and remove selected n samples from unlabelled
                to_add_df, unlabel_df = self.extract_features_and_compute_index(model = model, unlabel_df = unlabel_df, centroid = centroid, n = self.al_config.get("n"), outlier = self.al_config.get("outlier"), **self.model_config)
                label_df = label_df.append(to_add_df) # add selected n samples to labelled

            # phase 2: Training
            if model is not None:
                model = None
                torch.cuda.empty_cache()

            if current_step == 0:
                model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=None, model_config = self.model_config)
            else:
                model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=previous_model_path, model_config = self.model_config)
            print("Current Step:", current_step, " -- previous_model_path:", previous_model_path)
            
            # phase 3: Cluster Formation
            centroid = self.extract_features_and_form_clusters(model, label_df, **self.model_config)
            
            # phase 4: Evaluation
            result_df = self.evaluate(model, val_df, **self.model_config)
            
            # phase 5: Dump output
            self.dump_df(current_step_dir, label_df, to_add_df, unlabel_df, val_df) # dump samples
            self.dump_centroid(current_step_dir, centroid) # dump centroid
            self.dump_step_result(current_step_dir, result_df) # dump evaluation metrics

            # repeat
            current_step += 1

    def random_active_learning(self, full_df, val_df, random_state):
        model = None
        current_step = 0
        while current_step < self.max_steps:

            # phase 0: init step 0 directory 
            current_step_dir = self.create_step_dir(current_step)

            # phase 1: construct new label_df, unlabel_df, to_add_df 
            if current_step == 0:
                # Initialization and form df
                label_df, unlabel_df = self.construct_initial_training_set(full_df, **self.al_config)
                to_add_df = label_df.copy()
            else: 
                n = self.al_config["n"] 
                to_add_df = unlabel_df.sample(n, random_state = random_state)    
                unlabel_df = unlabel_df.drop(to_add_df.index)
                label_df = label_df.append(to_add_df) # add selected n samples to labelled        

            # phase 2: Training
            if model is not None:
                model = None
                torch.cuda.empty_cache()

            if current_step == 0:
                model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=None, model_config = self.model_config)
            else:
                model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=previous_model_path, model_config = self.model_config)
            print("Current Step:", current_step, " -- previous_model_path:", previous_model_path)
            
            # phase 3: Cluster Formation
            # centroid = self.extract_features_and_form_clusters(model, label_df, **self.model_config)
            
            # phase 4: Evaluation
            result_df = self.evaluate(model, val_df, **self.model_config)
            
            # phase 5: Dump output
            self.dump_df(current_step_dir, label_df, to_add_df, unlabel_df, val_df) # dump samples
            # self.dump_centroid(current_step_dir, centroid) # dump centroid
            self.dump_step_result(current_step_dir, result_df) # dump evaluation metrics

            # repeat
            current_step += 1

    def create_dir(self, main_dir, root_dir, **kwargs):
        '''A0: Create main directory for active learning output'''
        updated_main_dir = create_main_dir(f"{root_dir}/{main_dir}")
        self.al_config["updated_main_dir"] = updated_main_dir

    def create_step_dir(self, current_step):
        updated_main_dir = self.al_config["updated_main_dir"]
        current_step = str(current_step).zfill(3)
        current_step_dir = create_main_dir(f"{updated_main_dir}/step_{current_step}")
        return current_step_dir

    def construct_initial_training_set(self, full_df, m, n, random_state = 123, **kwargs):
        # m is the initial sample size
        # n is the subsequent resampling size
        label_df = full_df.sample(m, random_state = random_state)
        unlabel_df = full_df.drop(label_df.index).copy()
        while len(label_df["labels_x"].unique()) < 5:
            sub_df = unlabel_df.sample(n, random_state = random_state)
            label_df = label_df.append(sub_df)
            unlabel_df = unlabel_df.drop(sub_df.index).copy()
        return label_df, unlabel_df

    def train(self, current_step_dir, label_df, val_df, model, previous_model_path, model_config):
        model_config["main_model_dir"] = current_step_dir
        if previous_model_path is None:
            model_config["retrain"] = 0
            model_config["premodel_path"] = None   
        else:
            model_config["retrain"] = 1
            model_config["premodel_path"] = previous_model_path   

        CNN = ActiveDualPipeline(d_train = label_df, d_val = val_df, model = model, **model_config)
        model = CNN.get_model()
        best_model_path = CNN.get_best_model_path()
        return model, best_model_path

    def extract_features_and_form_clusters(self, model, label_df, size, batch_size, workers, load_only, **kwargs):
        data_loader = create_data_loader(df = label_df, size = size, batch_size = batch_size, workers = workers, debug_return_5 = 0, single_mode = 0, load_only = load_only)
        f, y = extract_features(model, data_loader)
        label_df["features"] = [j for j in f]

        centroid = {}
        for i in range(5):
            uf = label_df["labels_x"] == i
            sub_df = label_df.loc[uf, "features"]
            cent = sub_df.values.mean()
            centroid[i] = cent
            print(cent)
        return centroid

    def extract_features_and_compute_index(self, model, unlabel_df, centroid, n, outlier, size, batch_size, workers, load_only, **kwargs):
        data_loader = create_data_loader(df = unlabel_df, size = size, batch_size = batch_size, workers = workers, debug_return_5 = 0, single_mode = 0, load_only = load_only)
        f, y = extract_features(model, data_loader)
        ui_list = [unfamiliarity_index(feature, centroid) for feature in f]
        unlabel_df["ui"] = ui_list
        unlabel_df = unlabel_df.sort_values("ui", ascending = False)
        N = unlabel_df.shape[0]
        subN = int(N*(1-outlier))
        to_add_df = unlabel_df.tail(subN).head(n) # selected n samples
        unlabel_df = unlabel_df.drop(to_add_df.index)
        return to_add_df, unlabel_df

    def evaluate(self, model, val_df, size, batch_size, workers, load_only, **kwargs):
        data_loader = create_data_loader(df = val_df, size = size, batch_size = batch_size, workers = workers, debug_return_5 = 0, single_mode = 0, load_only = load_only)
        y_true, y_pred = predict(model, data_loader)
        acc = accuracy(y_true, y_pred)
        avg = avg_acc(y_true, y_pred)
        cm = get_cm(y_true, y_pred)
        qk = quadratic_kappa(y_true, y_pred)
        result = dict(accuracy = acc, average_accuracy = avg, kappa = qk, cm = cm)
        result_df = pd.DataFrame([result])
        return result_df

    def dump_df(self, current_step_dir, label_df, to_add_df, unlabel_df, val_df):
        label_df.to_csv(f"{current_step_dir}/label_df.csv")
        to_add_df.to_csv(f"{current_step_dir}/to_add_df.csv")
        unlabel_df.to_csv(f"{current_step_dir}/unlabel_df.csv")
        val_df.to_csv(f"{current_step_dir}/val_df.csv")

    def dump_centroid(self, current_step_dir, centroid):
        joblib.dump(centroid, f'{current_step_dir}/centroid.pkl')
        for i in range(5):
            centroid[i] = centroid[i].tolist()
            
        with open(f'{current_step_dir}/centroid.json', 'w') as outfile:
            json.dump(centroid, outfile)

    def dump_step_result(self, current_step_dir, result_df):
        if self.result_df is None:
            self.result_df = result_df
        else:
            self.result_df = self.result_df.append(result_df)
        self.result_df.to_csv(f"{current_step_dir}/result.csv")
    