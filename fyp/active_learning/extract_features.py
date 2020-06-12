import joblib
import numpy as np
import os
from tqdm import tqdm

def extract_features(model, unseen_test_loader):
    feature_extractor = model 
    feature_list = []
    y_list = []
    for i, (input, target) in tqdm(enumerate(unseen_test_loader), total=len(unseen_test_loader)):
        input, target = input.cuda(), target.cuda()
        output = feature_extractor(input) 
        x = output.cpu().detach().numpy().tolist()
        y = target.cpu().detach().numpy().tolist()
        feature_list += x
        y_list += y

    features = np.array(feature_list) #.reshape(-1, 8192)
    features = features.reshape((features.shape[0], -1))
    y_labels = np.array(y_list)
    print(features.shape, y_labels.shape)
    return features, y_labels
    # os.makedirs("data5", exist_ok =True)
    # os.makedirs(f"data5/{train_name}", exist_ok =True)

    # joblib.dump(features, filename = f"./{train_name}/X_{sample}.pkl")
    # joblib.dump(y_labels, filename = f"./{train_name}/y_{sample}.pkl")


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
    new_feature_list = [feature_list[i] + feature_list[-(i+1)] for i in range(k)]
    print(len(new_feature_list), k, len(new_feature_list[0]))

    features = np.array(new_feature_list) #.reshape(-1, 8192)
    features = features.reshape((features.shape[0], -1))
    y_labels = np.array(y_list[:k])
    print(features.shape, y_labels.shape)
    return features, y_labels