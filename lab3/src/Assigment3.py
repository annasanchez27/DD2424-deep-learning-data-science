from data.load_data import load_data,preprocessing,load_all_data,load_no_validation
from data.classifier import Classifier
import numpy as np
from utils import montage,error_plot,error_plot3,write_tofile,error_plot_normal
import math
from tqdm import tqdm



def check_matrices(a_anal,a_num):
    matrix = np.abs(a_anal - a_num)
    return np.amax(matrix)

def gradient_exercise(data):
    d = len(data['train_data']['data'][:, :10])
    m = 10
    k = len(data['train_data']['one_hot'])

    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k,input_nodes=m)

    print("Starting gradient exercise....")
    j_wrt_w1, j_wrt_b1,j_wrt_w2, j_wrt_b2 = classifier.compute_gradients(data['train_data']['data'][:, :20],
                                                    data['train_data']['one_hot'][:, :20],
                                                    'cross-entropy', 0)
    print(j_wrt_w1)
    print("Calculated my gradients!")
    classifier2 = Classifier()
    classifier2.add_layer(n=m, input_nodes=d)
    classifier2.add_layer(n=k,input_nodes=m)
    gradient_num = classifier2.ComputeGradsNum(data['train_data']['data'][:, :20],
                                              data['train_data']['one_hot'][:, :20],
                                              None, classifier2.layers[0].W, classifier2.layers[1].W,
                                              classifier2.layers[0].b,classifier2.layers[1].b, 0, 1e-6)

    print(gradient_num[0])
    print("gradient_w1 well calculated:", check_matrices(j_wrt_w1, gradient_num[0]))
    print("gradient_w2 well calculated:", check_matrices(j_wrt_w2, gradient_num[1]))
    print("gradient_b1 well calculated:", check_matrices(j_wrt_b1, gradient_num[2]))
    print("gradient_b2 well calculated:", check_matrices(j_wrt_b2, gradient_num[3]))


def check_convergence(data):
    d = len(data['train_data']['data'])
    m = len(data['train_data']['data'][0])
    k = len(data['train_data']['one_hot'])

    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    classifier.fit(data['train_data']['data'][:, :100],data['train_data']['one_hot'][:, :100],
                   data['train_data']['data'][:, :20],data['train_data']['one_hot'][:, :20],
                   data['train_data']['labels'],
                   'cross-entropy',
                   n_batch=100,eta=1e-5,n_epochs=200,lamda=0.01,eta_min=1e-5, eta_max=1e-1 ,stepsize=500,cycles=1)


def lerning_rate_exercise(data):
    d = len(data['train_data']['data'])
    m = 50
    k = len(data['train_data']['one_hot'])
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    metrics = classifier.fit(data['train_data']['data'],data['train_data']['one_hot'],
                   data['validation_data']['data'],data['validation_data']['one_hot'],
                   data['train_data']['labels'],
                   data['validation_data']['labels'],
                   'cross-entropy',
                   n_batch=100,eta=1e-5,n_epochs=10,lamda=0.01,eta_min=1e-5, eta_max=1e-1 ,n_s=500)
    error_plot(metrics['cost_train'],metrics['cost_val'],"Cost",2.5)
    error_plot(metrics['accuracy_train'], metrics['accuracy_val'], "Accuracy",60)
    error_plot(metrics['loss_train'], metrics['loss_val'], "Loss",2.5)
    _,prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'],prediction_test)
    print("Test accuracy:" ,test_accuracy)

def learning_rate_3cycles(data):
    d = len(data['train_data']['data'])
    m = 50
    k = len(data['train_data']['one_hot'])
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    metrics = classifier.fit(data['train_data']['data'],data['train_data']['one_hot'],
                   data['validation_data']['data'],data['validation_data']['one_hot'],
                   data['train_data']['labels'],
                   data['validation_data']['labels'],
                   'cross-entropy',
                   n_batch=100,eta=1e-5,n_epochs=48,lamda=0.01,eta_min=1e-5, eta_max=1e-1 ,n_s=800)
    error_plot3(metrics['cost_train'],metrics['cost_val'],"Cost")
    error_plot3(metrics['accuracy_train'], metrics['accuracy_val'], "Accuracy")
    error_plot3(metrics['loss_train'], metrics['loss_val'], "Loss")
    _,prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'],prediction_test)
    print("Test accuracy:" ,test_accuracy)

def broad_lambda_grid_search(data):
    n_batch = 100
    #Generate range of lambdas
    l_min = -6
    l_max = -1
    exponents_l = np.random.uniform(l_min,l_max,10)

    #Dimensions of the layers
    d = len(data['train_data']['data'])
    m = 50
    k = len(data['train_data']['one_hot'])
    n = len(data['train_data']['data'][0])

    for exp in tqdm(exponents_l):
        classifier = Classifier()
        classifier.add_layer(n=m, input_nodes=d)
        classifier.add_layer(n=k, input_nodes=m)
        metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                                 data['validation_data']['data'], data['validation_data']['one_hot'],
                                 data['train_data']['labels'],
                                 data['validation_data']['labels'],
                                 'cross-entropy',
                                 n_batch=n_batch, eta=1e-5, n_epochs=18, lamda=10**exp, eta_min=1e-5, eta_max=1e-1, n_s=2*math.floor(n/n_batch))

        write_tofile(10**exp,metrics['accuracy_val'][-1])


def narrower_lambda_grid_search(data):

    #Generate range of lambdas
    l_max = -4.5
    l_min = -5.5
    exponents_l = [0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]
    #Dimensions of the layers
    d = len(data['train_data']['data'])
    m = 50
    k = len(data['train_data']['one_hot'])
    n = len(data['train_data']['data'][0])

    n_batch = 90
    cycles = 2
    n_s = 1000
    n = len(data['train_data']['data'][0])
    print("N", n)
    epochs = int(cycles * n_s * 2 / (n / n_batch))
    print(epochs)

    for exp in tqdm(exponents_l):
        classifier = Classifier()
        classifier.add_layer(n=m, input_nodes=d)
        classifier.add_layer(n=k, input_nodes=m)
        metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                                 data['validation_data']['data'], data['validation_data']['one_hot'],
                                 data['train_data']['labels'],
                                 data['validation_data']['labels'],
                                 'cross-entropy',
                                 n_batch=n_batch, eta=1e-5, n_epochs=epochs, lamda=exp, eta_min=1e-5, eta_max=1e-1, n_s=n_s)
        print(exp)
        write_tofile(exp,metrics['accuracy_val'][-1])

def best_lambda_grid_search(data):
    d = len(data['train_data']['data'])
    m = 50
    k = len(data['train_data']['one_hot'])

    n_batch = 90
    cycles = 3
    n_s = 1000
    n = len(data['train_data']['data'][0])
    epochs = int(cycles*n_s*2/(n/n_batch))
    print(epochs)
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    metrics = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                             data['validation_data']['data'], data['validation_data']['one_hot'],
                             data['train_data']['labels'],
                             data['validation_data']['labels'],
                             'cross-entropy',
                             n_batch=n_batch, eta=1e-5, n_epochs=epochs, lamda=0.0009, eta_min=1e-5, eta_max=1e-1, n_s=n_s)
    #We will have 2000 ns
    error_plot_normal(metrics['loss_train'], metrics['loss_val'], "Loss")
    _,prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'],prediction_test)
    print("Test accuracy:" ,test_accuracy)


def main():
    #data = load_data()
    #data = preprocessing(data)
    #print("Preprocess data done!")
    #gradient_exercise(data)
    #check_convergence(data)
    #lerning_rate_exercise(data)
    #learning_rate_3cycles(data)

    #data = load_all_data()
    #data = preprocessing(data)
    #broad_lambda_grid_search(data)
    #narrower_lambda_grid_search(data)

    data = load_no_validation()
    data = preprocessing(data)
    best_lambda_grid_search(data)
if __name__ == "__main__":
    main()
