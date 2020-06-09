import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import adabound
from custom_math.kappa import quadratic_kappa

from pytorch.utils import torch_accuracy
from pytorch.utils import *
from pytorch.mnist import archs
import pytorch.metrics as metrics
from pytorch.coslr import CosineAnnealingWarmUpRestarts
from evaluate.metrics import avg_acc
from sklearn import metrics


# DEPRECATED
# def training_iterate(loader_data, model, metric_fc, criterion, losses, acc1s, metric):
#     y_pred = []
#     y_true = []
#     for i, (input, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
#         input = input.cuda()
#         target = target.long().cuda() 

#         feature = model(input)
#         if metric == 'softmax':
#             output = metric_fc(feature)
#         else:
#             output = metric_fc(feature, target)
#         loss = criterion(output, target)

#         acc1, = torch_accuracy(output, target, topk=(1,))
#         losses.update(loss.item(), input.size(0))
#         acc1s.update(acc1.item(), input.size(0))
        
#         y_pred += output.argmax(axis = 1).tolist()
#         y_true += target.tolist()
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     avg_acc_val = avg_acc(y_true, y_pred)   
#     qk = quadratic_kappa(y_true, y_pred)
#     return loss, qk #avg_acc_val


def logging(loss_avg, acc1_avg, avg_acc_val):
    log = OrderedDict([
        ('loss', loss_avg),
        ('acc_', acc1_avg),
        ('avg_acc_', avg_acc_val*100),
    ])
    return log


def train(loader_data, model, metric_fc, criterion, optimizer, metric = "not_softmax", batch_multiplier = 1):
    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to training mode
    model.train()
    metric_fc.train()

    # training
    # loss, avg_acc_val = training_iterate(train_loader, model, metric_fc, criterion, losses, acc1s, metric)
    y_pred = []
    y_true = []
    count = 0
    initial_loss = 0
    last_loss = 0
    for i, (input, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
        if count == 0:
            optimizer.step() # update cnn weights
            optimizer.zero_grad()
            count = batch_multiplier

        input, target = input.cuda(), target.long().cuda() 
        feature = model(input)
        if metric == 'softmax':
            output = metric_fc(feature)
        else:
            output = metric_fc(feature, target)

        loss = criterion(output, target)
        if i == 0:
            initial_loss = loss.item()
        else: 
            last_loss = loss.item()
        
        loss = loss / batch_multiplier
        loss.backward() # cumulate the loss 
        count -= 1

        # compute acc on the fly
        acc1, = torch_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))
        
        # record predicted
        y_pred += output.argmax(axis = 1).tolist()
        y_true += target.tolist()

    print("loss_0: ", initial_loss, "loss_n:", last_loss)
    # optimizer.step() # update cnn weights
    optimizer.zero_grad()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # calculate metrics using standard sklearn metrics
    # avg_acc_val = avg_acc(y_true, y_pred)   
    accuracy_val = metrics.accuracy_score(y_true, y_pred)
    qk = quadratic_kappa(y_true, y_pred)
    
    # log performance metrics
    log = logging(losses.avg, acc1s.avg, qk)
    return log


def validate(loader_data, model, metric_fc, criterion, metric = "not_softmax"):

    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to evaluate mode
    model.eval()
    metric_fc.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        for i, (input, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
            input = input.cuda()
            target = target.long().cuda() 

            feature = model(input)
            if metric == 'softmax':
                output = metric_fc(feature)
            else:
                output = metric_fc(feature, target)
            loss = criterion(output, target)

            acc1, = torch_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))

            y_pred += output.argmax(axis = 1).tolist()
            y_true += target.tolist()
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # avg_acc_val = avg_acc(y_true, y_pred)   
        qk = quadratic_kappa(y_true, y_pred)
        # loss, avg_acc_val = training_iterate(val_loader, model, metric_fc, criterion, losses, acc1s, metric)
    
    # log performance metrics
    log = logging(losses.avg, acc1s.avg, qk)
    return log


class PerformanceLog():
    def __init__(self):
        self.log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'train_loss', 'train_acc', 'train_avg_acc', 
            'val_loss', 'val_acc', 'val_avg_acc',
            'test_loss', 'test_acc', 'test_avg_acc',
        ])

    def append(self, epoch, lr, train_log, val_log, test_log):
        tmp = pd.Series([
            epoch,
            lr,
            train_log['loss'], train_log['acc_'], train_log['avg_acc_'],
            val_log['loss'], val_log['acc_'], val_log['avg_acc_'], 
            test_log['loss'], test_log['acc_'], test_log['avg_acc_']
        ], index=['epoch', 'lr', 'train_loss', 'train_acc', 'train_avg_acc',
                    'val_loss', 'val_acc', 'val_avg_acc',
                    'test_loss', 'test_acc', 'test_avg_acc'])
        self.log = self.log.append(tmp, ignore_index=True)

    def save(self, path):
        self.log.to_csv(path, index = False)