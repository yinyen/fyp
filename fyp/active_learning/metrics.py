import numpy as np

def unfamiliarity_index(feature, centroid_dict):
    ui = 0
    for key, val in centroid_dict.items():
        d = np.linalg.norm(feature-val)
        ui += np.sqrt(d)
    return ui