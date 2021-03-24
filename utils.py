import numpy as np


def one_hot_encoder(vector):
    n_values = np.max(vector) + 1
    b = np.eye(n_values)[vector]
    return b.T
