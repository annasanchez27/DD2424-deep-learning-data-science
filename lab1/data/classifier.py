import numpy as np
from lab1.src.utils import softmax
import random

class Classifier:

    def __init__(self,dim_images,num_labels):
        np.random.seed(400)
        self.W = np.random.normal(0, 0.01, size=(num_labels, dim_images))
        self.b = np.random.normal(0, 0.01, size=(num_labels, 1))

    def predict(self, X, loss_function):
        if loss_function == 'cross-entropy':
            return softmax(np.dot(self.W,X)+self.b)
        if loss_function == 'svm':
            return np.dot(self.W, X) + self.b


    def compute_cost(self,X,labels_onehot,prediction,loss_function,lambda_reg=0.0):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        entr = self._loss(labels_onehot, prediction,loss_function)
        return 1/num_datapoints*np.sum(entr)+ lambda_reg*np.sum(np.square(self.W))


    def _loss(self,label_onehotencoded,probabilities,loss_function):
        """Equation number (6) in the assigment"""
        if loss_function == 'cross-entropy':
            return -np.log(np.sum(label_onehotencoded * probabilities,axis=0))
        if loss_function == 'svm':
            s_y = np.sum(label_onehotencoded * probabilities, axis=0)
            s = np.maximum(np.zeros(probabilities.shape),probabilities-s_y+1)
            s[label_onehotencoded.astype(bool)] = 0
            return np.sum(s,axis=0)



    def compute_accuracy(self,data,prediction):
        """Equation number (4) in the assigment"""
        pred = np.argmax(prediction,axis=0)
        return np.sum(pred == data['labels'])/len(prediction[0])*100

    def compute_gradients(self,X,labels_onehot,predictions,loss_function,lambda_reg):
        """Equation (10) and (11) in the assigment. Look last slides Lecture 3"""
        n = X.shape[1]
        if loss_function == 'cross-entropy':
            g_batch = -(labels_onehot-predictions)
            j_wtr_w = 1/n*np.dot(g_batch,X.T) + 2*lambda_reg*self.W
            j_wrt_b = 1/n*np.dot(g_batch,np.ones((n,1)))
            return j_wtr_w,j_wrt_b

        if loss_function == 'svm':
            '''
            Loss gradient for SGD with batch size 1 only
            link to where I took the formula from: https://cs231n.github.io/optimization-1/
            '''
            s_y = np.sum(labels_onehot * predictions, axis=0)
            s = np.maximum(np.zeros(predictions.shape), predictions - s_y + 1)
            s[labels_onehot.astype(bool)] = 0
            vector = s
            vector[s > 0] = 1
            # count the number of classes that didn’t meet the desired margin
            num_misclass = np.sum(vector, axis=0)
            # in the right class we need to put the previous count
            vector[np.argmax(labels_onehot, axis=0) , np.arange(n)] = -num_misclass
            j_wtr_w = np.dot(vector, X.T) / n + lambda_reg * self.W
            j_wrt_b = 1/n*np.dot(vector,np.ones((n,1)))
            return j_wtr_w,j_wrt_b

    def fit(self,X_train,Y_train,X_val,Y_val,loss_function,n_batch,eta,n_epochs,lamda):
            n = X_train.shape[1]
            error_train =[]
            error_val = []
            for i in range(n_epochs):
                #indices = np.arange(Y_train.shape[1])
                #np.random.shuffle(indices)
                #X_train = X_train[:,indices]
                #Y_train = Y_train[:,indices]
                for j in range(int(n/n_batch)):
                    j = j + 1
                    j_start = (j - 1) * n_batch + 1
                    j_end = j*n_batch
                    X_batch = X_train[:, j_start:j_end]
                    Y_batch = Y_train[:,j_start:j_end]
                    prediction = self.predict(X_batch,loss_function)
                    j_wtr_w,j_wrt_b = self.compute_gradients(X_batch,Y_batch,prediction,loss_function,lamda)
                    self.W = self.W - eta*j_wtr_w
                    self.b = self.b - eta*j_wrt_b
                #eta = 0.9*eta
                prediction_train = self.predict(X_train,loss_function)
                cost_train = self.compute_cost(X_train, Y_train, prediction_train, loss_function,lamda)
                prediction_val = self.predict(X_val,loss_function)
                cost_val = self.compute_cost(X_val, Y_val, prediction_val, loss_function,lamda)
                print("Epoch #" + str(i) + " Loss:" + str(cost_val))
                error_train.append(cost_train)
                error_val.append(cost_val)

            return {'loss_train':error_train,
                    'loss_val':error_val}


    def ComputeCost(self, X, Y, W, b, lamda):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        prediction = self.Predict(X, W, b)
        entr = self._loss(Y, prediction,'cross-entropy')
        return 1 / num_datapoints * np.sum(entr) + lamda * np.sum(np.square(W))

    def Predict(self, X, W, b):
        return softmax(np.dot(W, X) + b)

    def ComputeGradsNum(self,X, Y, P, W, b, lamda, h):
        """ Converted from matlab code """
        no = W.shape[0]
        d = X.shape[0]

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros((no, 1));

        c = self.ComputeCost(X, Y, W, b,lamda);

        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(X, Y, W, b_try,lamda)
            grad_b[i] = (c2 - c) / h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i, j] += h
                c2 = self.ComputeCost(X, Y, W_try, b, lamda)
                grad_W[i, j] = (c2 - c) / h

        return [grad_W, grad_b]
