import pandas as pd
import numpy as np
import torch
from pytorch.param_helper import create_dir, create_main_dir, import_config
from active_learning.model_pipeline import TrainPipeline
from active_learning.data_gen import create_data_loader
from active_learning.extract_features import extract_features
import glob
import json
import joblib
from tqdm import tqdm
from evaluate.metrics import accuracy, avg_acc, get_cm
from custom_math.kappa import quadratic_kappa
import torch.utils.model_zoo as model_zoo

def reset_model(model):
    model.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'))
    return model

def unfamiliarity_index(feature, centroid_dict):
    ui = 0
    for key, val in centroid_dict.items():
        d = np.linalg.norm(feature-val)
        ui += np.sqrt(d)
    return ui

class ActiveLearning():
    def __init__(self, **config):
        self.result_df = None
        self.result_df2 = None
        self.config = config
        self.max_steps = self.config.get("max_steps")

        self.create_dir(**self.config) #create main dir, return updated_main_dir (increment version if exist)
        full_df, initial_d_unlabel = self.temp_get_filenames(self.config.get("main_data_dir"), train_dir = "full_train")
        test_df, _ = self.temp_get_filenames(self.config.get("main_data_dir"), train_dir = "val")

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
        centroid = self.extract_features_and_form_clusters(model, label_df, **config)

        # phase 2: active learning
        while current_step < self.max_steps:
            if current_step > 0:
                # init new step
                current_step_dir = self.create_step_dir(current_step)

                # re-train model
                # model, metric_fc = self.train(current_step_dir, label_df, label_df, model, metric_fc, **self.config) # currently only train and validate on label_df
                reset_model(model)
                model, metric_fc = self.train(current_step_dir, label_df, val_df, model, metric_fc, **self.config) # currently only train and validate on label_df
                
                # extract features and update clusters
                centroid = self.extract_features_and_form_clusters(model, label_df, **config)

                # add selected n samples to labelled
                label_df = label_df.append(val_df)
        

            # compute unfamiliarity index and remove selected n samples from unlabelled
            val_df, unlabel_df = self.extract_features_and_compute_index(model, unlabel_df, centroid, **self.config)
            
            # # add selected n samples to labelled
            # label_df = label_df.append(val_df)
        
            # evaluate model on selected n samples
            result_df = self.evaluate(model, metric_fc, val_df, **self.config)
            result_df2 = self.evaluate(model, metric_fc, test_df, **self.config)

            # dump results
            # dump label_df, val_df, unlabel_df and dump centroid
            self.dump_df(current_step_dir, label_df, val_df, unlabel_df)
            self.dump_centroid(current_step_dir, centroid)
            # dump evaluation metrics
            self.dump_step_result(current_step_dir, result_df)
            self.dump_step_result2(current_step_dir, result_df2)

            # repeat
            current_step += 1
    
    def create_dir(self, main_dir, root_dir, **kwargs):
        updated_main_dir = create_main_dir(f"{root_dir}/{main_dir}")
        self.config["updated_main_dir"] = updated_main_dir

    def temp_get_filenames(self, main_data_dir, train_dir = "full_train", **kwargs):
        f1 = glob.glob(f"{main_data_dir}/{train_dir}/*/*.jpeg")
        # f2 = glob.glob(f"{main_data_dir}/val/*/*.jpeg")
        # all_files = f1 + f2
        all_files = f1
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

    def extract_features_and_form_clusters(self, model, label_df, size, workers, **kwargs):
        data_loader = create_data_loader(label_df, size, batch_size = 6, workers = workers)
        f, y = extract_features(model, data_loader)
        label_df["features"] = [j for j in f]

        centroid = {}
        for i in range(5):
            uf = label_df["labels"] == str(i)
            cent = label_df.loc[uf, "features"].values.mean()
            centroid[i] = cent
        return centroid

    def extract_features_and_compute_index(self, model, unlabel_df, centroid, n, size, workers, outlier = 0, **kwargs):
        data_loader = create_data_loader(unlabel_df, size, batch_size = 6, workers = workers)
        f, y = extract_features(model, data_loader)
        ui_list = [unfamiliarity_index(feature, centroid) for feature in f]
        unlabel_df["ui"] = ui_list
        unlabel_df = unlabel_df.sort_values("ui", ascending = False)
        N = unlabel_df.shape[0]
        subN = int(N*(1-outlier))
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
    
    def dump_step_result2(self, current_step_dir, result_df2):
        if self.result_df2 is None:
            self.result_df2 = result_df2
        else:
            self.result_df2 = self.result_df2.append(result_df2)
        self.result_df2.to_csv(f"{current_step_dir}/result_2.csv")