from lab2.data.load_data import load_data,preprocessing
from lab2.data.classifier import Classifier
import numpy as np
from utils import montage,error_plot,error_plot3




def check_matrices(a_anal,a_num):
    matrix = np.abs(a_anal - a_num)
    return np.isin(matrix <= 1e-6, True).all()

def gradient_exercise(data):
    d = len(data['train_data']['data'][:, :10])
    m = len(data['train_data']['data'][:, :10][0])
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
    m = 10
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
    error_plot(metrics['cost_train'],metrics['cost_val'],"Cost")
    error_plot(metrics['accuracy_train'], metrics['accuracy_val'], "Accuracy")
    error_plot(metrics['loss_train'], metrics['loss_val'], "Loss")
    _,prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'],prediction_test)
    print("Test accuracy:" ,test_accuracy)

def learning_rate_3cycles(data):
    d = len(data['train_data']['data'])
    m = 10
    k = len(data['train_data']['one_hot'])
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k, input_nodes=m)
    metrics = classifier.fit(data['train_data']['data'],data['train_data']['one_hot'],
                   data['validation_data']['data'],data['validation_data']['one_hot'],
                   data['train_data']['labels'],
                   data['validation_data']['labels'],
                   'cross-entropy',
                   n_batch=100,eta=1e-5,n_epochs=50,lamda=0.01,eta_min=1e-5, eta_max=1e-1 ,n_s=800)
    error_plot3(metrics['cost_train'],metrics['cost_val'],"Cost")
    error_plot3(metrics['accuracy_train'], metrics['accuracy_val'], "Accuracy")
    error_plot3(metrics['loss_train'], metrics['loss_val'], "Loss")
    _,prediction_test = classifier.predict(data['test_data']['data'], 'cross-entropy')
    test_accuracy = classifier.compute_accuracy(data['test_data']['labels'],prediction_test)
    print("Test accuracy:" ,test_accuracy)

def main():
    data = load_data()
    data = preprocessing(data)
    print("Preprocess data done!")
    #gradient_exercise(data)
    #check_convergence(data)
    #lerning_rate_exercise(data)
    learning_rate_3cycles(data)

if __name__ == "__main__":
    main()
