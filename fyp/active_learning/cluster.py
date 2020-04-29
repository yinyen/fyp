import numpy as np
import pandas as pd
from custom_math.dist import euclidean_distance

def compute_centroid_dict(features, labels):
    # input: features and labels of the LABELLED samples 
    features_flatten = features.reshape((features.shape[0],-1))

    centroid_dict = {}
    for LABEL in range(5):
        p1 = labels == LABEL
        subset_features = features_flatten[p1]
        if len(subset_features) > 0:
            centroid = subset_features.mean(axis = 0) # calculate the centroid for label == LABEL
            centroid_dict[LABEL] = centroid

    return centroid_dict

def compute_distance_df_per_batch(predicted_features, filenames, centroid_dict):
    flattened_features = predicted_features.reshape((predicted_features.shape[0], -1))
    distance_list = []
    for feature, filename in zip(flattened_features, filenames):
        dist={}
        unfamiliarity_index = 0
        for key, val in centroid_dict.items():
            d = euclidean_distance(feature, val)
            dist[key] = d
            unfamiliarity_index += np.sqrt(d)
        dist["unfamiliarity_index"] = unfamiliarity_index
        dist["img_file"] = filename
        distance_list.append(dist)
    df = pd.DataFrame(distance_list)
    return df