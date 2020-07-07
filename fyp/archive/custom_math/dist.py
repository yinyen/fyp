import numpy as np


def euclidean_distance(x, y):
    dist = np.linalg.norm(x-y)
    return dist
