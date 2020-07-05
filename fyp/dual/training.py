import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from custom_math.kappa import quadratic_kappa
from evaluate.metrics import avg_acc, get_cm
from pytorch.utils import torch_accuracy, AverageMeter

def count_unique(x, x2):
    for j in np.unique(x):
        y = np.sum(x == j)
        y2 = np.sum(x2 == j)
        print("Count {}: actual, pred - {}, {}".format(j, y, y2))

def convert_pred(x):
    if x < 0.5:
        return 0 
    elif x < 1.5:
        return 1
    elif x < 2.5:
        return 2
    elif x < 3.5:
        return 3
    else:
        return 4 

def train(loader_data, model, criterion, optimizer, batch_multiplier = 1):
    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to training mode
    model.train()

    # training
    y_pred = []
    y_true = []
    count = 0
    initial_loss = 0
    last_loss = 0
    for i, (input1, input2, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
        if count == 0:
            optimizer.step() # update cnn weights
            optimizer.zero_grad()
            count = batch_multiplier

        input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
        output = model(input1, input2)
        output = output.flatten()
        loss = criterion(output, target)
        if i == 0:
            initial_loss = loss.item()
        else: 
            last_loss = loss.item()
        
        loss = loss / batch_multiplier
        loss.backward() # cumulate the loss 
        count -= 1

        # compute acc on the fly
        # acc1, = torch_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), target.size(0))
        # acc1s.update(acc1.item(), target.size(0))
        
        # record predicted 
        # y_pred += output.argmax(axis = 1).tolist() # 5 neurons
        to_add = output.flatten().tolist()
        to_add2 = [int(convert_pred(j)) for j in to_add]
        y_pred += to_add2 # 1 continuous neuron
        y_true += target.int().tolist()

    print("loss_0: ", initial_loss, "loss_n:", last_loss)
    # optimizer.step() # update cnn weights
    optimizer.zero_grad()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # calculate metrics using standard sklearn metrics
    accuracy_val = metrics.accuracy_score(y_true, y_pred)*100
    avg_acc_val = avg_acc(y_true, y_pred)*100
    qk = quadratic_kappa(y_true, y_pred)*100
    
    # log performance metrics
    log = {"qk": qk, "loss": losses.avg, "acc": accuracy_val, "avg_acc": avg_acc_val}
    return log


def validate(loader_data, model, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        for i, (input1, input2, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
            input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
            output = model(input1, input2)
            output = output.flatten()
            loss = criterion(output, target)

            # acc1, = torch_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), target.size(0))
            # acc1s.update(acc1.item(), target.size(0))

            # y_pred += output.argmax(axis = 1).tolist() # 5 neurons
            to_add = output.flatten().tolist()
            to_add2 = [int(convert_pred(j)) for j in to_add]
            y_pred += to_add2 # 1 continuous neuron
            y_true += target.int().tolist()
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # calculate metrics using standard sklearn metrics
        print("true:", y_true)
        print("pred:", y_pred)
        cm = get_cm(y_true, y_pred)
        print("val_cm:", cm)
        count_unique(y_true, y_pred)

        accuracy_val = metrics.accuracy_score(y_true, y_pred)*100
        avg_acc_val = avg_acc(y_true, y_pred)*100
        qk = quadratic_kappa(y_true, y_pred)*100

    # log performance metrics
    log = {"qk": qk, "loss": losses.avg, "acc": accuracy_val, "avg_acc": avg_acc_val}
    return log


class PerformanceLog():
    def __init__(self):
        self.log = pd.DataFrame()

    def append(self, epoch, lr, train_log, val_log, test_log):
        train_log = train_log.copy()
        val_log = val_log.copy()
        test_log = test_log.copy()

        new_keys = [("train_" + key, key) for key in train_log.keys()]
        for new_key, key in new_keys:
            train_log[new_key] = train_log.pop(key)

        new_keys = [("val_" + key, key) for key in val_log.keys()]
        for new_key, key in new_keys:
            val_log[new_key] = val_log.pop(key)

        new_keys = [("test_" + key, key) for key in test_log.keys()]
        for new_key, key in new_keys:
            test_log[new_key] = test_log.pop(key)

        full_log = {**train_log, **val_log, **test_log}
        full_log["epoch"] = epoch
        full_log["lr"] = lr
        # print(full_log)

        tmp = pd.Series(full_log)
        self.log = self.log.append(tmp, ignore_index=True)

    def save(self, path):
        self.log.to_csv(path, index = False)