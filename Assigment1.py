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
    mean_x = np.mean(data['train_data']['data'],axis=1)
    mean_x = mean_x[:, np.newaxis]
    std_x = np.std(data['train_data']['data'],axis=1)
    std_x = std_x[:,np.newaxis]
    data['train_data']['data'] = (data['train_data']['data'] - mean_x)/std_x
    data['test_data']['data'] = (data['test_data']['data'] - mean_x) / std_x
    data['validation_data']['data'] = (data['validation_data']['data'] - mean_x) / std_x
    return data


def check_matrices(a_anal,a_num):
    matrix = np.abs(a_anal - a_num)
    return np.isin(matrix <= 1e-6, True).all()



def main():
    data = load_data()
    data = preprocessing(data)
    classifier = Classifier(dim_images=len(data['train_data']['data']),
                            num_labels=len(data['train_data']['one_hot']))
    """
    prediction = classifier.predict(data['train_data']['data'])
    cost = classifier.compute_cost(data['train_data']['data'],data['train_data']['one_hot'],prediction)
    print("Computed cost:", cost)
    accuracy = classifier.compute_accuracy(data['train_data'],prediction)
    print("Computed accuracy:", accuracy)
    j_wrt_w,j_wrt_b = classifier.compute_gradients(data['train_data']['data'][:, :20],data['train_data']['one_hot'][:, :20],prediction[:, :20],0)
    gradient_num = classifier.ComputeGradsNum(data['train_data']['data'][:, :20],data['train_data']['one_hot'][:, :20],prediction,classifier.W,classifier.b,0,1e-6)
    print("gradient_w well calculated:",check_matrices(j_wrt_w,gradient_num[0]))
    print("gradient_b well calculated:", check_matrices(j_wrt_b, gradient_num[1]))
    """
    classifier.mini_batch(data['train_data']['data'],data['train_data']['one_hot'],n_batch=100,eta=0.001,n_epochs=40,lamda=0)
    prediction_test = classifier.predict(data['test_data']['data'])
    acc_test = classifier.compute_accuracy(data['test_data'],prediction_test)
    print("Accuracy in test set:", acc_test)

if __name__ == "__main__":
    main()

