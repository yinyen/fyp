import joblib
import numpy as np
import os
from tqdm import tqdm
import torch
from torch import nn
from dual.training import convert_pred

def extract_features(model, loader_data):
    features_list = []
    features_extractor = nn.Sequential(*list(model.children())[:-1])
    for i, (input1, input2, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
        input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
        output = features_extractor(input1) 
        features_list.append(np.array(output.tolist()))

    features_extractor = None
    torch.cuda.empty_cache()

    features = np.concatenate(features_list, axis=0)
    features = features.reshape((features.shape[0], features.shape[1]))

    return features, None


def predict(model, loader_data):
    y_true = []
    y_pred = []
    for i, (input1, input2, target) in tqdm(enumerate(loader_data), total=len(loader_data)):
        input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
        output = model(input1) 

        # ypred = output.cpu().detach().numpy().tolist()
        # ytrue = target.cpu().detach().numpy().tolist()
        to_add = output.flatten().tolist()
        to_add2 = [int(convert_pred(j)) for j in to_add]
        y_pred += to_add2 # 1 continuous neuron
        y_true += target.int().tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(y_true.shape, y_pred.shape)
    return y_true, y_pred


def extract_dual_features(model, unseen_test_loader):
    feature_extractor = model 
    feature_list = []
    y_list = []
    for i, (input1, input2, target) in tqdm(enumerate(unseen_test_loader), total=len(unseen_test_loader)):
        input1, input2, target = input1.cuda(), input2.cuda(), target.cuda()
        output = feature_extractor(input1, input2) 
        x = output.cpu().detach().numpy().tolist()
        y = target.cpu().detach().numpy().tolist()
        feature_list += x
        y_list += y

    N = len(y_list)
    k = N // 2
    new_feature_list = [feature_list[i] + feature_list[k+1] for i in range(k)]
    print(len(new_feature_list), k, len(new_feature_list[0]))

    features = np.array(new_feature_list) #.reshape(-1, 8192)
    features = features.reshape((features.shape[0], -1))
    y_labels = np.array(y_list[:k])
    print(features.shape, y_labels.shape)
    return features, y_labels


def indices_to_one_hot(data, nb_classes = 5):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def extract_xy(model, data_gen, one_hot = False):
    features_list = []
    y_true = []
    features_extractor = nn.Sequential(*list(model.children())[:-1])
    for i, (input1, input2, target) in tqdm(enumerate(data_gen), total=len(data_gen)):
        input1, input2, target = input1.cuda(), input2.cuda(), target.float().cuda() 
        output = features_extractor(input1) 
        np_output = np.array(output.tolist())
        features_list.append(np_output)
        target_to_add = target.int().tolist()
        y_true += target_to_add
    # features = np.array(features_list)
    features = np.concatenate(features_list, axis=0)
    features = features.reshape((features.shape[0], features.shape[1]))
    y_true = np.array(y_true)
    if one_hot:
        y_true = indices_to_one_hot(y_true)
    return features, y_true
