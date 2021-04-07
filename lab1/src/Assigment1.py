from data.load_data import load_data,preprocessing
from data.classifier import Classifier
import numpy as np
from utils import montage,error_plot




def check_matrices(a_anal,a_num):
    matrix = np.abs(a_anal - a_num)
    return np.isin(matrix <= 1e-7, True).all()


def gradient_exercise(classifier,data):
    prediction = classifier.predict(data['train_data']['data'],'cross-entropy')
    cost = classifier.compute_cost(data['train_data']['data'], data['train_data']['one_hot'], prediction,'cross-entropy')
    print("Computed cost:", cost)
    accuracy = classifier.compute_accuracy(data['train_data'], prediction)
    print("Computed accuracy:", accuracy)
    j_wrt_w, j_wrt_b = classifier.compute_gradients(data['train_data']['data'][:, :20],
                                                    data['train_data']['one_hot'][:, :20], prediction[:, :20],'cross-entropy' ,0)
    gradient_num = classifier.ComputeGradsNum(data['train_data']['data'][:, :20], data['train_data']['one_hot'][:, :20],
                                              prediction, classifier.W, classifier.b, 0, 1e-6)
    print("gradient_w well calculated:", check_matrices(j_wrt_w, gradient_num[0]))
    print("gradient_b well calculated:", check_matrices(j_wrt_b, gradient_num[1]))

def main():
    data = load_data(exc2=True)
    data = preprocessing(data)
    print("Preprocessing done!")
    classifier = Classifier(dim_images=len(data['train_data']['data']),
                            num_labels=len(data['train_data']['one_hot']))
    gradient_exercise(classifier,data)
    loss = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                          data['validation_data']['data'], data['validation_data']['one_hot'],
                          loss_function='cross-entropy',
                          n_batch=80, eta=0.01, n_epochs=10, lamda=0)
    error_plot(loss['loss_train'], loss['loss_val'])
    prediction_test = classifier.predict(data['test_data']['data'], loss_function='cross-entropy')
    montage(classifier.W)
    acc_test = classifier.compute_accuracy(data['test_data'], prediction_test)
    print("Accuracy in test set:", acc_test)



if __name__ == "__main__":
    main()

