import numpy as np

def unfamiliarity_index(feature, centroid_dict):
    ui = 0
    for key, val in centroid_dict.items():
        d = np.linalg.norm(feature-val)
        ui += np.sqrt(d)
    return ui

def unfamiliarity_index_with_all(feature, X_label):
    ui = 0
    n = X_label.shape[0]
    for val in X_label:
        d = np.linalg.norm(feature-val)
        ui += np.sqrt(d)
    ui = ui / n
    return ui