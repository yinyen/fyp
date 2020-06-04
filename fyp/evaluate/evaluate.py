import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from collections import Counter


def sort_df(x):
    k = list(Counter(x).keys())
    v = list(Counter(x).values())
    df =pd.DataFrame(dict(key = k, count = v)).sort_values("key")
    df["prop"] = df["count"]/df["count"].sum()
    return df


def sort_df(x):
    k = list(Counter(x).keys())
    v = list(Counter(x).values())
    df =pd.DataFrame(dict(key = k, count = v)).sort_values("key")
    df["prop"] = df["count"]/df["count"].sum()
    return df
