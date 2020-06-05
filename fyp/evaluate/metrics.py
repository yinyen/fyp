import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from evaluate.evaluate import *


def accuracy(true, pred):
    acc = np.mean(true == pred)
    return acc

def get_dist(true, pred):
    print("distribution: true")
    print(sort_df(true))
    print("distribution: prediction")
    print(sort_df(pred))

def get_cm(true, pred):
    cm = confusion_matrix(true, pred, normalize='true')
    return pd.DataFrame(cm)
    
def avg_acc(true, pred):
    cm = confusion_matrix(true, pred, normalize='true')
    n = cm.shape[0]
    return sum([cm[i,i] for i in range(n)])/n
