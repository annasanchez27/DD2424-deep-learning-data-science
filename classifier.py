import numpy as np
from utils import softmax

class Classifier:

    def __init__(self,dim_images,num_labels):
        self.W = np.random.normal(0, 0.01, size=(num_labels, dim_images))
        self.b = np.random.normal(0, 0.01, size=(num_labels, 1))

    def predict(self, X):
        return softmax(np.dot(self.W,X)+self.b)

    def compute_cost(self,X,labels_onehot,prediction,lambda_reg=1.0):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        entr = self._cross_entropy(labels_onehot, prediction)
        return 1/num_datapoints*np.sum(entr)+ lambda_reg*np.sum(np.square(self.W))


    def _cross_entropy(self,label_onehotencoded,probabilities):
        """Equation number (6) in the assigment"""
        r = -np.log(np.sum(label_onehotencoded * probabilities,axis=0))
        return r

    def compute_accuracy(self,data,prediction):
        """Equation number (4) in the assigment"""
        pred = np.argmax(prediction,axis=0)
        return np.sum(pred == data['labels'])/len(prediction[0])*100

    def compute_gradients(self,X,labels_onehot,predictions,lambda_reg):
        """Equation (10) and (11) in the assigment. Look last slides Lecture 3"""
        n = X.shape[1]
        g_batch = -(labels_onehot-predictions)
        j_wtr_w = 1/n*np.dot(g_batch,X.T) + 2*lambda_reg*self.W
        j_wrt_b = 1/n*np.dot(g_batch,np.ones((n,1)))
        return j_wtr_w,j_wrt_b

    def ComputeCost(self, X, Y, W, b, lamda):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        prediction = self.Predict(X, W, b)
        entr = self._cross_entropy(Y, prediction)
        return 1 / num_datapoints * np.sum(entr) + lamda * np.sum(np.square(W))

    def Predict(self, X, W, b):
        return softmax(np.dot(W, X) + b)

    def ComputeGradsNum(self,X, Y, P, W, b, lamda, h):
        """ Converted from matlab code """
        no = W.shape[0]
        d = X.shape[0]

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros((no, 1));

        c = self.ComputeCost(X, Y, W, b, lamda);

        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(X, Y, W, b_try, lamda)
            grad_b[i] = (c2 - c) / h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i, j] += h
                c2 = self.ComputeCost(X, Y, W_try, b, lamda)
                grad_W[i, j] = (c2 - c) / h

        return [grad_W, grad_b]