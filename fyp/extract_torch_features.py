# CUDA_VISIBLE_DEVICES=1 python extract_torch_features.py
import torch
from tqdm import tqdm
from pytorch_adacos.mnist import archs
from pytorch_adacos.metrics import AdaCos
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from pytorch_adacos.model_helper import select_model, select_metric
import joblib
import numpy as np
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

main_model_dir = "torch_models" #torch_models
model_type = "xception"
train_name = "xception_new_warm_v1_v02"
best = "best_acc_"
batch_size = 16

metric_type = "adacos"
model=select_model(model_type, {})

PATH = f"./{main_model_dir}/{train_name}/{best}model.pth"
PATH2 = f"./{main_model_dir}/{train_name}/{best}metric_fc.pth"

metric_fc = select_metric(metric_type, num_ftr = 1000)

model.load_state_dict(torch.load(PATH))
metric_fc.load_state_dict(torch.load(PATH2))
model.eval()
metric_fc.eval()

def extract_save_features(model, train_name, sample):
    transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    unseen_test_set = datasets.ImageFolder(root=f'./data3/{sample}', transform=transform_test)
    unseen_test_loader = torch.utils.data.DataLoader(
            unseen_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8)
            
    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = model #torch.nn.Sequential(*list(model.children()))

    feature_list = []
    y_list = []
    for i, (input, target) in tqdm(enumerate(unseen_test_loader), total=len(unseen_test_loader)):
        input, target = input.to(device), target.to(device)
        output = feature_extractor(input) 
        x = output.cpu().detach().numpy().tolist()
        y = target.cpu().detach().numpy().tolist()
        feature_list += x
        y_list += y

    features = np.array(feature_list) #.reshape(-1, 8192)
    features = features.reshape((features.shape[0], -1))
    y_labels = np.array(y_list)
    print(features.shape, y_labels.shape)

    os.makedirs("data5", exist_ok =True)
    os.makedirs(f"data5/{train_name}", exist_ok =True)

    joblib.dump(features, filename = f"data5/{train_name}/X_{sample}.pkl")
    joblib.dump(y_labels, filename = f"data5/{train_name}/y_{sample}.pkl")

extract_save_features(model, train_name, "train")
extract_save_features(model, train_name, "val")
extract_save_features(model, train_name, "test")
# print("Done")
