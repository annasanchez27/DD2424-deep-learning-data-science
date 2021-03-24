import pickle
import numpy as np
from utils import one_hot_encoder

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_batch(file):
    dict = unpickle(file)
    one_hot = one_hot_encoder(dict[b'labels'])
    return {'data': dict[b'data'],
            'labels': dict[b'labels'],
            'one_hot': one_hot}


n_dict = load_batch("cifar-10-batches-py/data_batch_1")
