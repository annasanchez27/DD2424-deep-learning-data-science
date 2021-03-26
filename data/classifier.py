import numpy as np
from src.utils import softmax

class Classifier:

    def __init__(self,dim_images,num_labels):
        np.random.seed(400)
        self.W = np.random.normal(0, 0.01, size=(num_labels, dim_images))
        self.b = np.random.normal(0, 0.01, size=(num_labels, 1))

    def predict(self, X):
        return softmax(np.dot(self.W,X)+self.b)

    def compute_cost(self,X,labels_onehot,prediction,lambda_reg=0.0):
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



    def mini_batch(self,X_train,Y_train,X_val,Y_val,n_batch,eta,n_epochs,lamda):
        n = X_train.shape[1]
        error_train =[]
        error_val = []
        for _ in range(n_epochs):
            for j in range(int(n/n_batch)):
                j = j + 1
                j_start = (j - 1) * n_batch + 1
                j_end = j*n_batch
                X_batch = X_train[:, j_start:j_end]
                Y_batch = Y_train[:,j_start:j_end]
                prediction = self.predict(X_batch)
                j_wtr_w,j_wrt_b = self.compute_gradients(X_batch,Y_batch,prediction,lamda)
                self.W = self.W - eta*j_wtr_w
                self.b = self.b - eta*j_wrt_b

            prediction_train = self.predict(X_train)
            cost_train = self.compute_cost(X_train, Y_train, prediction_train, lamda)
            prediction_val = self.predict(X_val)
            cost_val = self.compute_cost(X_val, Y_val, prediction_val, lamda)
            error_train.append(cost_train)
            error_val.append(cost_val)

        return {'loss_train':error_train,
                'loss_val':error_val}


    #Analytical functions
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
