import pickle
import numpy as np
from lab1.src.utils import one_hot_encoder

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

def load_data(exc2=True):
    if exc2:
        train_data1 = load_batch("../cifar-10-batches-py/data_batch_1")
        train_data2 = load_batch("../cifar-10-batches-py/data_batch_2")
        train_data3 = load_batch("../cifar-10-batches-py/data_batch_3")
        train_data4 = load_batch("../cifar-10-batches-py/data_batch_4")
        train_data5 = load_batch("../cifar-10-batches-py/data_batch_5")
        data = np.concatenate((train_data1['data'], train_data2['data'], train_data3['data'],
                                 train_data4['data'],train_data5['data'][:, :9000]), axis=1)

        labels = np.concatenate([train_data1['labels'], train_data2['labels'], train_data3['labels'],
                                 train_data4['labels'],train_data5['labels'][:9000]])

        one_hot = np.concatenate((train_data1['one_hot'], train_data2['one_hot'], train_data3['one_hot'],
                                 train_data4['one_hot'],train_data5['one_hot'][:, :9000]), axis=1)
        train_data = {'data': data, 'labels': labels, 'one_hot': one_hot}

        validation_data = {'data': train_data5['data'][:,9000:],
                           'labels':train_data5['labels'][9000:],
                           'one_hot': train_data5['one_hot'][:,9000:]}
        test_data = load_batch("../cifar-10-batches-py/test_batch")
    else:
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
