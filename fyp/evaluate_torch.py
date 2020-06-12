# CUDA_VISIBLE_DEVICES=1 python evaluate_torch.py
import torch
from tqdm import tqdm
from pytorch.mnist import archs
from pytorch.metrics import AdaCos
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from pytorch.model_helper import select_model, select_metric
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

main_model_dir = "torch_models" #torch_models
main_model_dir = "/media/workstation/Storage/Test/AL/al_val_v01" #torch_models
model_type = "xception"
train_name = "step_048"
best = "best_avg_acc_"
best = "best_acc_"
size = 400
batch_size = 6

metric_type = "softmax"
model=select_model(model_type, {})

PATH = f"{main_model_dir}/{train_name}/{best}model.pth"
PATH2 = f"{main_model_dir}/{train_name}/{best}metric_fc.pth"

metric_fc = select_metric(metric_type, num_ftr = 1000, num_classes=5)

model.load_state_dict(torch.load(PATH))
metric_fc.load_state_dict(torch.load(PATH2))
model.eval()
metric_fc.eval()

transform_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
unseen_test_set = datasets.ImageFolder(root='../all_train_300/test', transform=transform_test)
unseen_test_loader = torch.utils.data.DataLoader(
        unseen_test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

y_true = []
y_pred = []
for i, (input, target) in tqdm(enumerate(unseen_test_loader), total=len(unseen_test_loader)):
    input, target = input.to(device), target.to(device)
    # input, target2 = input.cuda(), target.cuda()
    feature = model(input)
    if metric_type=="softmax":
        output = metric_fc(feature)
    else:
        output = metric_fc(feature, target)
    y_pred += output.cpu().detach().numpy().argmax(axis = 1).tolist()
    y_true += target.cpu().numpy().tolist()
y_true = np.array(y_true)
y_pred = np.array(y_pred)
len(y_pred), len(y_true)

from evaluate.metrics import *
from custom_math.kappa import quadratic_kappa
acc = accuracy(y_true, y_pred)
avg = avg_acc(y_true, y_pred)
cm = get_cm(y_true, y_pred)
qk = quadratic_kappa(y_true, y_pred)
get_dist(y_true, y_pred)
print("Accuracy:", acc)
print("Avg Accuracy:", avg)
print("QK:", qk)
print(cm)
