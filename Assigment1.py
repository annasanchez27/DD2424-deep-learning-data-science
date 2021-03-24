from load_data import load_batch
from classifier import Classifier
import numpy as np

def load_data():
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    validation_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")

    return {'train_data':train_data,
            'validation_data': validation_data,
            'test_data': test_data}

def preprocessing(data):
    mean_x = np.mean(data['train_data']['data'],axis=0)
    std_x = np.std(data['train_data']['data'],axis=0)
    data['train_data']['data'] = (data['train_data']['data'] - mean_x)/std_x
    data['test_data']['data'] = (data['test_data']['data'] - mean_x) / std_x
    data['validation_data']['data'] = (data['validation_data']['data'] - mean_x) / std_x
    return data

def main():
    data = load_data()
    data = preprocessing(data)
    classifier = Classifier(dim_images=len(data['train_data']['data']),
                            num_labels=len(data['train_data']['one_hot']))

    prediction = classifier.predict(data['train_data']['data'])
    cost = classifier.compute_cost(data['train_data'],prediction)
    print(cost)
    accuracy = classifier.compute_accuracy(data['train_data'],prediction)
    print(accuracy)

if __name__ == "__main__":
    main()

