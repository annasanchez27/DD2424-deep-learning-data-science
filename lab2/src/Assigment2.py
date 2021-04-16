from lab2.data.load_data import load_data,preprocessing,load_all_data,load_no_validation
from lab2.data.classifier import Classifier
import numpy as np
from utils import montage,error_plot,error_plot3,write_tofile,error_plot_normal,plot_accuracies_numbernodes
import math
from tqdm import tqdm
import matplotlib.pyplot as plt






def hiddenodes_grid_search(data):
    #Generate range of lambdas
    l_max = -4.5
    l_min = -5.5
    exponents_l = [10,50,200,400,800,2000]
    for m in tqdm(exponents_l):
        #Dimensions of the layers
        d = len(data['train_data']['data'])
        k = len(data['train_data']['one_hot'])
        n = len(data['train_data']['data'][0])
        n_batch = 90
        cycles = 2
        n_s = 1000
        n = len(data['train_data']['data'][0])
        print("N", n)
        epochs = int(cycles * n_s * 2 / (n / n_batch))
        print(epochs)
        classifier = Classifier()
        classifier.add_layer(n=m, input_nodes=d)
        classifier.add_layer(n=k, input_nodes=m)
        metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                                 data['validation_data']['data'], data['validation_data']['one_hot'],
                                 data['train_data']['labels'],
                                 data['validation_data']['labels'],
                                 'cross-entropy',
                                 n_batch=n_batch, eta=1e-5, n_epochs=epochs, lamda=0.0005, eta_min=1e-5, eta_max=1e-1, n_s=n_s)
        print(m)
        write_tofile(m,metrics['accuracy_val'][-1])

def dropout(data):
    l_max = -4.5
    l_min = -5.5
    dropout = [True,False]
    acc = []
    for dr in tqdm(dropout):
        # Dimensions of the layers
        d = len(data['train_data']['data'])
        k = len(data['train_data']['one_hot'])
        n = len(data['train_data']['data'][0])
        m = 750
        n_batch = 90
        cycles = 3
        n_s = 1000
        n = len(data['train_data']['data'][0])
        epochs = int(cycles * n_s * 2 / (n / n_batch))
        classifier = Classifier()
        classifier.add_layer(n=m, input_nodes=d)
        classifier.add_layer(n=k, input_nodes=m)
        metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                                 data['validation_data']['data'], data['validation_data']['one_hot'],
                                 data['train_data']['labels'],
                                 data['validation_data']['labels'],
                                 'cross-entropy',
                                 n_batch=n_batch, eta=1e-5, n_epochs=epochs, lamda=0.0001, eta_min=1e-5, eta_max=1e-1,
                                 n_s=n_s,dropout=dr)
        acc.append(metrics['accuracy_val'])
        plt.plot(metrics['accuracy_val'],label="Dropout: "+str(dr))
    plt.legend()
    plt.show()


def jitter(data):

    l_max = -4.5
    l_min = -5.5
    jitter = [True, False]
    acc = []
    for jit in tqdm(jitter):
        # Dimensions of the layers
        d = len(data['train_data']['data'])
        k = len(data['train_data']['one_hot'])
        n = len(data['train_data']['data'][0])
        m = 750
        n_batch = 90
        cycles = 3
        n_s = 1000
        n = len(data['train_data']['data'][0])
        epochs = int(cycles * n_s * 2 / (n / n_batch))
        classifier = Classifier()
        classifier.add_layer(n=m, input_nodes=d)
        classifier.add_layer(n=k, input_nodes=m)
        metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                                 data['validation_data']['data'], data['validation_data']['one_hot'],
                                 data['train_data']['labels'],
                                 data['validation_data']['labels'],
                                 'cross-entropy',
                                 n_batch=n_batch, eta=1e-5, n_epochs=epochs, lamda=0.0001, eta_min=1e-5,
                                 eta_max=1e-1,
                                 n_s=n_s, dropout=False,jitter=jit)
        acc.append(metrics['accuracy_val'])
        plt.plot(metrics['accuracy_val'], label="Jitter: " + str(jit))
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()

def best_model(data):
    # Dimensions of the layers
    d = len(data['train_data']['data'])
    k = len(data['train_data']['one_hot'])
    n = len(data['train_data']['data'][0])
    m = 750
    n_batch = 90
    cycles = 3
    n_s = 1000
    n = len(data['train_data']['data'][0])
    epochs = int(cycles * n_s * 2 / (n / n_batch))
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                             data['validation_data']['data'], data['validation_data']['one_hot'],
                             data['train_data']['labels'],
                             data['validation_data']['labels'],
                             'cross-entropy',
                             n_batch=n_batch, eta=1e-5, n_epochs=epochs, lamda=0.001, eta_min=1e-5,
                             eta_max=1e-1,
                             n_s=n_s, dropout=False,jitter=False)
    _,prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'],prediction_test)
    print("Test accuracy:" ,test_accuracy)

def learning_rate(data):

    # Dimensions of the layers
    d = len(data['train_data']['data'])
    k = len(data['train_data']['one_hot'])
    n = len(data['train_data']['data'][0])
    m = 750
    n_batch = 90
    cycles = 3
    n_s = 1000
    n = len(data['train_data']['data'][0])
    epochs = int(cycles * n_s * 2 / (n / n_batch))
    print(epochs)
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                             data['validation_data']['data'], data['validation_data']['one_hot'],
                             data['train_data']['labels'],
                             data['validation_data']['labels'],
                             'cross-entropy',
                             n_batch=n_batch, eta=0, n_epochs=epochs, lamda=0.0001, eta_min=0,
                             eta_max=0.02,
                             n_s=n_s, dropout=False,jitter=False)

    _, prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'], prediction_test)
    print("Test accuracy:", test_accuracy)

    error_plot(metrics['cost_train'], metrics['cost_val'], "Cost", epochs)
    error_plot(metrics['accuracy_train'], metrics['accuracy_val'], "Accuracy", epochs)
    error_plot(metrics['loss_train'], metrics['loss_val'], "Loss", epochs)

def main():
    data = load_no_validation()
    data = preprocessing(data)
    #hiddenodes_grid_search(data)
    #plot_accuracies_numbernodes()
    #dropout(data)
    #jitter(data)
    #best_model(data)
    learning_rate(data)


if __name__ == "__main__":
    main()
