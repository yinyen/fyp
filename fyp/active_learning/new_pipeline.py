import os
import time
import glob
import json
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial import distance

import torch
import torch.backends.cudnn as cudnn
from pytorch.param_helper import create_dir, create_main_dir, import_config, dump_config
from dual.dual_df_helper import create_dual_label_df, split_dual_df
from dual.dual_gen_helper import initialize_dual_gen, create_data_loader
from dual.pipeline import DualPipeline
from active_learning.extract_features import extract_features, predict, extract_xy
from active_learning.metrics import unfamiliarity_index
from evaluate.metrics import accuracy, avg_acc, get_cm
from evaluate.kappa import quadratic_kappa
from pytorch.model_helper import select_model
from sklearn.linear_model import LogisticRegression


class ActiveDualPipeline(DualPipeline):
    def __init__(self, d_train, d_val, model, **kwargs):
        t0 = time.time()
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
        t1 = time.time()
        self.total_time = (t1-t0)//60
        print("Total time taken: {} minutes".format(self.total_time))
        self.dump_time(self.model_path)
        
    def init_dataset(self, d_train, d_val, batch_size, size, workers, reweight_sample = 1, reweight_sample_factor = 2, single_mode = 0, load_only = 0, **kwargs):
        # initialize dataset generators
        print("Train:", d_train.shape, "Val:", d_val.shape)
        train_gen, val_gen = initialize_dual_gen(d_train, d_val, size, batch_size, reweight_sample, reweight_sample_factor, workers, single_mode = 0, load_only = load_only) # force double image for now
        return train_gen, val_gen


class ActiveLearning():
    def __init__(self, **config):
        self.result_df = None
        self.result_df2 = None

        self.cn_config = config.get("cn_config")
        self.al_config = config.get("al_config")
        self.model_config = config.get("model_config")
        self.max_steps = self.al_config.get("max_steps")

        self.style = self.al_config.get("style")
        self.override_reset_model = self.al_config.get("override_reset_model")

        ## Continue restoration
        current_step, packet, main_path = self.continue_restoration(self.cn_config, self.al_config, self.model_config) 

        # A0: Create Active Learning directory
        if current_step == 0:
            self.create_dir(**self.al_config) # create main dir, return updated_main_dir (increment version if exist)
        else:
            self.al_config["updated_main_dir"] = main_path

        ## Dump config
        output_path = os.path.join(self.al_config["updated_main_dir"], "active_learning_config.yaml")
        dump_config(output_path, config)

        # A1: Partition 
        dual_df = create_dual_label_df(main_data_dir = self.model_config.get("main_data_dir"), train_dir_list = self.model_config.get("train_dir_list"))

        full_df, test_df = split_dual_df(dual_df, p = None, seed = self.al_config.get("random_state"), n = self.al_config.get("test_n_sample")) 
        full_df, val_df = split_dual_df(full_df, p = None, seed = self.al_config.get("random_state"), n = self.al_config.get("val_n_sample")) 
        print("full, val, test:", full_df.shape, val_df.shape, test_df.shape)
        full_df["ui"] = 0
        full_df["features"] = 0

        self.full_df = full_df
        self.val_df = val_df
        self.test_df = test_df

        # A2: Active Learning --- # full_df, current_step, centroid, unlabel_df, model, val_df, previous_model_path
        # Loop
        self.loop(current_step, packet)

    def continue_restoration(self, cn_config, al_config, model_config):
        main_path = cn_config.get("main_path")
        current_step = cn_config.get("current_step")
        previous_step_path = cn_config.get("previous_step_path")
        style = al_config.get("style")
        train_name = model_config.get("train_name")
        model_type = model_config.get("model_type")
        model_kwargs = model_config.get("model_kwargs")

        if current_step == 0:
            return current_step, None, main_path

        if style == "ui":
            label_df = pd.read_csv(f"{previous_step_path}/label_df.csv")
            unlabel_df = pd.read_csv(f"{previous_step_path}/unlabel_df.csv")
            previous_model_path = f'{previous_step_path}/{train_name}/best_qk_model.pth'
            model = select_model(model_type, model_kwargs)
            model.load_state_dict(torch.load(previous_model_path))
            centroid = joblib.load(f'{previous_step_path}/centroid.pkl')
            packet = (label_df, unlabel_df, model, previous_model_path, centroid)
        elif style in ["random", "maxentropy", "maxentropy_dist", "maxentropy_ui", "maxentropy_can"]:
            label_df = pd.read_csv(f"{previous_step_path}/label_df.csv")
            unlabel_df = pd.read_csv(f"{previous_step_path}/unlabel_df.csv")
            previous_model_path = f'{previous_step_path}/{train_name}/best_qk_model.pth'
            model = select_model(model_type, model_kwargs)
            model.load_state_dict(torch.load(previous_model_path))
            packet = (label_df, unlabel_df, model, previous_model_path)
        print("Continuing from step: ", current_step)

        return current_step, packet, main_path

    def loop(self, current_step, packet):
        # model = None
        # current_step = 0
        while current_step < self.max_steps:
            if self.al_config.get("style") == "ui":
                packet = self.ui_active_learning(current_step, packet)
            elif self.al_config.get("style") == "random":
                packet = self.random_active_learning(current_step, packet, self.al_config.get("random_state"))
            elif self.al_config.get("style") in ["maxentropy","maxentropy_dist","maxentropy_ui", "maxentropy_can"]:
                packet = self.maxentropy_active_learning(current_step, packet)

            current_step += 1 

    def ui_active_learning(self, current_step, packet):
        t0 = time.time()
        val_df = self.val_df
        test_df = self.test_df

        # phase -1: continue restoration
        if packet is not None:
            label_df, unlabel_df, model, previous_model_path, centroid = packet #unpack
        else:
            model = None

        # phase 0: init step 0 directory    
        current_step_dir = self.create_step_dir(current_step)

        # phase 1: construct new label_df, unlabel_df, to_add_df 
        if current_step == 0:
            # Initialization and form df
            full_df = self.full_df
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

        label_num_samples = label_df.shape[0]
        unlabel_num_samples = unlabel_df.shape[0]
        if current_step == 0:
            model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=None, model_config = self.model_config)
        else:
            model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=previous_model_path, model_config = self.model_config)
        print("Current Step:", current_step, " -- previous_model_path:", previous_model_path)
        
        # phase 3: Cluster Formation
        centroid = self.extract_features_and_form_clusters(model, label_df, **self.model_config)
        
        # phase 4: Evaluation
        result_df = self.evaluate(model, val_df, **self.model_config)
        result_df2 = self.evaluate(model, test_df, **self.model_config)
        t1 = time.time()

        # phase 5: Dump output
        self.dump_df(current_step_dir, label_df, to_add_df, unlabel_df, val_df) # dump samples
        self.dump_centroid(current_step_dir, centroid) # dump centroid

        result_df["label_num_samples"] = label_num_samples
        result_df["unlabel_num_samples"] = unlabel_num_samples
        result_df["time"] = (t1-t0)/60
        self.dump_step_result(current_step_dir, result_df) # dump evaluation metrics
        self.dump_step_result2(current_step_dir, result_df2) # dump evaluation metrics
        self.dump_time(current_step_dir, t0, t1)
        
        # repeat
        # current_step += 1
        packet = (label_df, unlabel_df, model, previous_model_path, centroid)
        return packet

    def maxentropy_active_learning(self, current_step, packet):
        t0 = time.time()
        val_df = self.val_df
        test_df = self.test_df

        # phase -1: continue restoration
        if packet is not None:
            label_df, unlabel_df, model, previous_model_path = packet #unpack
        else:
            model = None

        # phase 0: init step 0 directory    
        current_step_dir = self.create_step_dir(current_step)

        # phase 1: construct new label_df, unlabel_df, to_add_df 
        if current_step == 0:
            # Initialization and form df
            full_df = self.full_df
            label_df, unlabel_df = self.construct_initial_training_set(full_df, **self.al_config)
            to_add_df = label_df.copy()
        else: 
            # compute entropy for each unlabel, and select top n entropy 
            to_add_df, unlabel_df = self.entropy_selection(model = model, label_df = label_df, unlabel_df = unlabel_df, n = self.al_config.get("n"), outlier = self.al_config.get("outlier"), **self.model_config)
            label_df = label_df.append(to_add_df) # add selected n samples to labelled

        # phase 2: Training
        if model is not None:
            model = None
            torch.cuda.empty_cache()

        label_num_samples = label_df.shape[0]
        unlabel_num_samples = unlabel_df.shape[0]
        if current_step == 0:
            model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=None, model_config = self.model_config)
        else:
            model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=previous_model_path, model_config = self.model_config)
        print("Current Step:", current_step, " -- previous_model_path:", previous_model_path)
        
        # phase 3: Cluster Formation
        # centroid = self.extract_features_and_form_clusters(model, label_df, **self.model_config)
        
        # phase 4: Evaluation
        result_df = self.evaluate(model, val_df, **self.model_config)
        result_df2 = self.evaluate(model, test_df, **self.model_config)
        t1 = time.time()

        # phase 5: Dump output
        self.dump_df(current_step_dir, label_df, to_add_df, unlabel_df, val_df) # dump samples
        # self.dump_centroid(current_step_dir, centroid) # dump centroid
        result_df["time"] = (t1-t0)/60
        result_df["label_num_samples"] = label_num_samples
        result_df["unlabel_num_samples"] = unlabel_num_samples
        self.dump_step_result(current_step_dir, result_df) # dump evaluation metrics
        self.dump_step_result2(current_step_dir, result_df2) # dump evaluation metrics
        self.dump_time(current_step_dir, t0, t1)
        
        # repeat
        # current_step += 1
        packet = (label_df, unlabel_df, model, previous_model_path)
        return packet


    def random_active_learning(self, current_step, packet, random_state):
        t0 = time.time()
        val_df = self.val_df
        test_df = self.test_df

        # phase -1: continue restoration
        if packet is not None:
            label_df, unlabel_df, model, previous_model_path = packet #unpack
        else:
            model = None
            
        # phase 0: init step 0 directory 
        current_step_dir = self.create_step_dir(current_step)

        # phase 1: construct new label_df, unlabel_df, to_add_df 
        if current_step == 0:
            # Initialization and form df
            full_df = self.full_df
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

        label_num_samples = label_df.shape[0]
        unlabel_num_samples = unlabel_df.shape[0]
        if current_step == 0:
            model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=None, model_config = self.model_config)
        else:
            model, previous_model_path = self.train(current_step_dir=current_step_dir, label_df=label_df, val_df=val_df, model=model, previous_model_path=previous_model_path, model_config = self.model_config)
        print("Current Step:", current_step, " -- previous_model_path:", previous_model_path)
            
        # phase 3: Cluster Formation
        # centroid = self.extract_features_and_form_clusters(model, label_df, **self.model_config)
        
        # phase 4: Evaluation
        result_df = self.evaluate(model, val_df, **self.model_config)
        result_df2 = self.evaluate(model, test_df, **self.model_config)
        t1 = time.time()

        # phase 5: Dump output
        self.dump_df(current_step_dir, label_df, to_add_df, unlabel_df, val_df) # dump samples
        # self.dump_centroid(current_step_dir, centroid) # dump centroid
        result_df["label_num_samples"] = label_num_samples
        result_df["unlabel_num_samples"] = unlabel_num_samples
        result_df["time"] = (t1-t0)/60
        self.dump_step_result(current_step_dir, result_df) # dump evaluation metrics
        self.dump_step_result2(current_step_dir, result_df2) # dump evaluation metrics
        self.dump_time(current_step_dir, t0, t1)

        # repeat
        # current_step += 1
        packet = label_df, unlabel_df, model, previous_model_path
        return packet


    def create_dir(self, main_dir, root_dir, **kwargs):
        '''A0: Create main directory for active learning output'''
        updated_main_dir = create_main_dir(f"{root_dir}/{main_dir}")
        self.al_config["updated_main_dir"] = updated_main_dir

    def create_step_dir(self, current_step):
        updated_main_dir = self.al_config["updated_main_dir"]
        current_step = str(current_step).zfill(3)
        current_step_dir = create_main_dir(f"{updated_main_dir}/step_{current_step}")
        return current_step_dir

    def construct_initial_training_set(self, full_df, m, n, random_state = 123, demo = 0, **kwargs):
        # m is the initial sample size
        # n is the subsequent resampling size
        if demo:
            full_df = full_df.sample(600, random_state = random_state)
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

        if self.override_reset_model == 1:
            model_config["retrain"] = 0
            model_config["premodel_path"] = None   

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
    
    def entropy_selection(self, model, label_df, unlabel_df, n, outlier, size, batch_size, reweight_sample, reweight_sample_factor, workers, load_only, **kwargs):
        train_gen, unlabel_gen = initialize_dual_gen(label_df, unlabel_df, size = size, batch_size = batch_size, reweight_sample = reweight_sample, reweight_sample_factor = reweight_sample_factor, workers = workers, single_mode = 0, load_only = load_only)

        # extract features
        X_train, y_train = extract_xy(model = model, data_gen = train_gen)
        print("(entropy) Label/Train shape:", X_train.shape, y_train.shape)
        X_unlabel, y_unlabel = extract_xy(model = model, data_gen = unlabel_gen)
        print("(entropy) Unlabel shape:", X_unlabel.shape, y_unlabel.shape)

        # train logistic
        clf = LogisticRegression(random_state=0, max_iter=100).fit(X_train, y_train)
        
        # predict entropy
        # y_pred = clf.predict(X_unlabel)
        probs = clf.predict_proba(X_unlabel)
        entp = probs * np.log(probs)
        entropy = -1*np.sum(entp, axis = 1)
        unlabel_df["entropy"] = entropy
        
        if self.style == "maxentropy":
            unlabel_df = unlabel_df.sort_values("entropy", ascending = False)
            to_add_df = unlabel_df.head(n) # selected n samples
            
        elif self.style == "maxentropy_dist":
            ## + distance function
            avg_dist_list = []
            for x in X_unlabel:
                q = x - X_train
                dist = np.linalg.norm(q, axis = 1)
                avg_dist = np.mean(dist)
                avg_dist_list.append(avg_dist)
            unlabel_df["avg_dist"] = avg_dist_list
            unlabel_df = unlabel_df.sort_values("entropy", ascending = False)
            to_add_df = unlabel_df.head(100) # selected n samples
            to_add_df = to_add_df.sort_values("avg_dist", ascending = False).head(n)

        elif self.style == "maxentropy_ui":
            ## + distance function
            avg_dist_list = []
            for x in X_unlabel:
                q = x - X_train
                dist = np.sqrt(np.linalg.norm(q, axis = 1))
                avg_dist = np.mean(dist)
                avg_dist_list.append(avg_dist)
            unlabel_df["avg_dist"] = avg_dist_list
            unlabel_df = unlabel_df.sort_values("entropy", ascending = False)
            to_add_df = unlabel_df.head(100) # selected n samples
            to_add_df = to_add_df.sort_values("avg_dist", ascending = False).head(n)
        
        elif self.style == "maxentropy_can":
            ## + distance function
            avg_dist_list = []
            for x in X_unlabel:
                cd_list = [distance.canberra(x, y) for y in X_train]   
                avg_dist = np.mean(cd_list)
                avg_dist_list.append(avg_dist)
            unlabel_df["avg_dist"] = avg_dist_list
            unlabel_df = unlabel_df.sort_values("entropy", ascending = False)
            to_add_df = unlabel_df.head(100) # selected n samples
            to_add_df = to_add_df.sort_values("avg_dist", ascending = False).head(n)

        # remove outlier, select top n
        # N = unlabel_df.shape[0]
        # subN = int(N*(1-outlier))
        # to_add_df = unlabel_df.tail(subN).head(n) # selected n samples
        unlabel_df = unlabel_df.drop(to_add_df.index)

        print("=============END ENTROPY===========")
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

    def dump_step_result2(self, current_step_dir, result_df_test):
        if self.result_df2 is None:
            self.result_df2 = result_df_test
        else:
            self.result_df2 = self.result_df2.append(result_df_test)
        self.result_df2.to_csv(f"{current_step_dir}/result_test.csv")
    
    def dump_time(self, current_step_dir, t0, t1):
        total_time = (t1 - t0)//60
        output_path = os.path.join(current_step_dir, "time.yaml")
        time_kw = {"time_taken": total_time}
        dump_config(output_path, time_kw)