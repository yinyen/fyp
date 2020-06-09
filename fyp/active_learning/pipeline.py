import pandas as pd
import numpy as np
import torch
from pytorch.param_helper import create_dir, create_main_dir, import_config
from active_learning.model_pipeline import TrainPipeline
from active_learning.data_gen import create_data_loader
from active_learning.extract_features import extract_features
import glob
from tqdm import tqdm
from evaluate.metrics import accuracy, avg_acc, get_cm
from custom_math.kappa import quadratic_kappa

def unfamiliarity_index(feature, centroid_dict):
    ui = 0
    for key, val in centroid_dict.items():
        d = np.linalg.norm(feature-val)
        ui += np.sqrt(d)
    return ui

class ActiveLearning():
    def __init__(self, **config):
        self.result_df = None
        self.config = config
        self.create_dir(**self.config) #create main dir, return updated_main_dir (increment version if exist)
        full_df, initial_d_unlabel = self.temp_get_filenames(**self.config)
        full_df["ui"] = 0
        full_df["features"] = 0

        current_step = 0
        
        ## STEP 0:
        # phase 0: init step j directory and config
        current_step_dir = self.create_step_dir(current_step)
        if current_step == 0:
            label_df, val_df, unlabel_df = self.construct_initial_training_set(full_df, initial_d_unlabel, **config)

        # phase 1: Formation of initial cluster
        model, metric_fc = self.train(current_step_dir, label_df, val_df, None, None, **self.config)
        centroid = self.extract_features_and_form_initial_clusters(model, label_df, **config)

        # phase 2: active learning
        while current_step < 3:
            if current_step > 0:
                # init new step
                current_step_dir = self.create_step_dir(current_step)

                # re-train model
                model, metric_fc = self.train(current_step_dir, label_df, label_df, model, metric_fc, **self.config) # currently only train and validate on label_df

                # extract features and update clusters
                # centroid = self.extract_features_and_update_clusters(model, label_df, **config)

            print("pre-added:", label_df.shape)
            print("pre-added:", unlabel_df.shape)

            # compute unfamiliarity index and remove selected n samples from unlabelled
            val_df, unlabel_df = self.extract_features_and_compute_index(model, unlabel_df, centroid, **self.config)
            
            # add selected n samples to labelled
            label_df = label_df.append(val_df)
            print("added:", label_df.shape)
            print("added:", unlabel_df.shape)
        
            # evaluate model on selected n samples
            # metrics = self.evaluate(model, metric_fc, val_df)

            # dump results
            # dump label_df, val_df, unlabel_df
            self.dump_df(current_step_dir, label_df, val_df, unlabel_df)
           
            # dump evaluation metrics
            result_df = self.evaluate(model, metric_fc, val_df, **self.config)
            self.dump_step_result(current_step_dir, result_df)

            # repeat
            current_step += 1
    
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
        # val_df = unlabel_df.sample(n, random_state = random_state) # val_df is for next step training, but used for this step validation
        # unlabel_df = unlabel_df.drop(val_df.index).copy()
        val_df = label_df.copy()
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

    def extract_features_and_compute_index(self, model, unlabel_df, centroid, n, size, workers, **kwargs):
        data_loader = create_data_loader(unlabel_df, size, batch_size = 6, workers = workers)
        f, y = extract_features(model, data_loader)
        ui_list = [unfamiliarity_index(feature, centroid) for feature in f]
        unlabel_df["ui"] = ui_list
        unlabel_df = unlabel_df.sort_values("ui", ascending = False)
        N = unlabel_df.shape[0]
        subN = int(N*0.98)
        toadd_df = unlabel_df.tail(subN).head(n) # selected n samples
        unlabel_df = unlabel_df.drop(toadd_df.index)
        return toadd_df, unlabel_df

    def train(self, current_step_dir, label_df, val_df, model, metric_fc, **kwargs):
        CNN = TrainPipeline(step_dir = current_step_dir, label_df = label_df, val_df = val_df, model = model, metric_fc = metric_fc, **kwargs)
        model, metric_fc = CNN.get_model()
        return model, metric_fc

    def evaluate(self, model, metric_fc, val_df, size, workers, metric_type, **kwargs):
        unseen_test_loader = create_data_loader(val_df, size, batch_size = 6, workers = workers)
        y_true = []
        y_pred = []
        for i, (input, target) in tqdm(enumerate(unseen_test_loader), total=len(unseen_test_loader)):
            input, target = input.cuda(), target.cuda()
            feature = model(input)
            if metric_type=="softmax":
                output = metric_fc(feature)
            else:
                output = metric_fc(feature, target)
            y_pred += output.cpu().detach().numpy().argmax(axis = 1).tolist()
            y_true += target.cpu().numpy().tolist()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        acc = accuracy(y_true, y_pred)
        avg = avg_acc(y_true, y_pred)
        cm = get_cm(y_true, y_pred)
        qk = quadratic_kappa(y_true, y_pred)
        result = dict(accuracy = acc, average_accuracy = avg, kappa = qk, cm = cm)
        result_df = pd.DataFrame([result])
        return result_df

    def dump_df(self, current_step_dir, label_df, val_df, unlabel_df):
        label_df.to_csv(f"{current_step_dir}/label_df.csv")
        val_df.to_csv(f"{current_step_dir}/selected_df.csv")
        unlabel_df.to_csv(f"{current_step_dir}/unlabel_df.csv")

    def dump_step_result(self, current_step_dir, result_df):
        if self.result_df is None:
            self.result_df = result_df
        else:
            self.result_df = self.result_df.append(result_df)
        self.result_df.to_csv(f"{current_step_dir}/result.csv")