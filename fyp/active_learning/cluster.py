import numpy as np
import pandas as pd
from custom_math.dist import euclidean_distance
from preprocessing.load import load_img_fast, get_label_from_filename

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

def predict_and_compute_distance_in_batch(fmodel, all_filenames, centroid_dict, output_folder, num_per_batch = 256, LIMIT=None):
    if LIMIT is not None:
        all_filenames = np.random.choice(all_filenames, size = LIMIT, replace = False)
    n = num_per_batch
    j = 0
    start = j*n
    end = (j+1)*n
    df_list = []
    while start < len(all_filenames):
        x = load_img_fast(all_filenames[start:end])
        predicted_features = fmodel.predict(x)
        df = compute_distance_df_per_batch(predicted_features, all_filenames[start:end], centroid_dict)

        j = j+1
        start = j*n
        end = (j+1)*n
        df.to_csv(f'{output_folder}/{j}.csv')
        df_list.append(df)
        del x
        del predicted_features

    full_df = pd.concat(df_list)
    full_df.to_csv(f'{output_folder}/full.csv')

    return full_df


def construct_new_training_set(training_set, full_dist_df, label_df, remove_top = 0.02, EVERY_SAMPLE = 20):
    p1 = full_dist_df["img_file"].isin(training_set["img_file"])
    unlabelled_df = full_dist_df.loc[~p1].sort_values("unfamiliarity_index", ascending = False)
    j = int(unlabelled_df.shape[0]*remove_top) + 1
    clean_unlabelled_df = unlabelled_df.iloc[j:,:]
    to_label = clean_unlabelled_df.head(EVERY_SAMPLE).copy()

    to_label["label"] = to_label["img_file"].apply(lambda x: get_label_from_filename(x, label_df))
    to_label = to_label[["label", "img_file"]]
    new_training_set = pd.concat([training_set, to_label])
    return new_training_set
