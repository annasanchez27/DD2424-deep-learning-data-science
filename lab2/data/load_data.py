import pickle
import numpy as np
from lab2.src.utils import one_hot_encoder

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_batch(file):
    dict = unpickle(file)
    one_hot = one_hot_encoder(dict[b'labels'])
    return {'data': dict[b'data'].T,
            'labels': dict[b'labels'],
            'one_hot': one_hot}


def load_data():
    train_data = load_batch("../cifar-10-batches-py/data_batch_1")
    validation_data = load_batch("../cifar-10-batches-py/data_batch_2")
    test_data = load_batch("../cifar-10-batches-py/test_batch")

    return {'train_data':train_data,
            'validation_data': validation_data,
            'test_data': test_data}



def preprocessing(data):
    mean_x = np.mean(data['train_data']['data'],axis=1)
    mean_x = mean_x[:, np.newaxis]
    std_x = np.std(data['train_data']['data'],axis=1)
    std_x = std_x[:,np.newaxis]
    data['train_data']['data'] = (data['train_data']['data'] - mean_x)/std_x
    data['test_data']['data'] = (data['test_data']['data'] - mean_x) / std_x
    data['validation_data']['data'] = (data['validation_data']['data'] - mean_x) / std_x
    return data
