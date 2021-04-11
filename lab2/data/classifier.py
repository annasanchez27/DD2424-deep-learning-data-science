import numpy as np
from lab2.src.utils import softmax
import random
from lab2.data.layer import Layer
from tqdm import tqdm


class Classifier:
    def __init__(self):
        self.layers = []

    def add_layer(self,n,input_nodes):
        layer = Layer(n,input_nodes)
        self.layers.append(layer)


    def predict_layer(self,X,W,b,activation_function):
        if activation_function == 'softmax':
            return softmax(np.dot(W,X)+b)
        if activation_function == 'relu':
            return np.maximum(np.zeros(shape=(np.dot(W, X) + b).shape), np.dot(W, X) + b)


    def predict(self,X,loss_function):
        h = self.predict_layer(X,self.layers[0].W, self.layers[0].b, 'relu')
        prediction = self.predict_layer(h, self.layers[1].W, self.layers[1].b, 'softmax')
        return h,prediction

    def compute_cost(self,X,labels_onehot,prediction,loss_function,lambda_reg):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        entr = self._loss(labels_onehot, prediction,loss_function)
        return 1/num_datapoints*np.sum(entr)+ lambda_reg* (np.sum(self.layers[0].W**2) + np.sum(self.layers[1].W))


    def _loss(self,label_onehotencoded,probabilities,loss_function):
        """Equation number (6) in the assigment"""
        if loss_function == 'cross-entropy':
            return -np.log(np.sum(label_onehotencoded * probabilities,axis=0))
        if loss_function == 'svm':
            s_y = np.sum(label_onehotencoded * probabilities, axis=0)
            s = np.maximum(np.zeros(probabilities.shape),probabilities-s_y+1)
            s[label_onehotencoded.astype(bool)] = 0
            return np.sum(s,axis=0)



    def compute_accuracy(self,labels,prediction):
        """Equation number (4) in the assigment"""
        pred = np.argmax(prediction,axis=0)
        return np.sum(pred == labels )/len(prediction[0])*100

    def compute_gradients(self,X,labels_onehot,loss_function,lambda_reg):
        """Equation (10) and (11) in the assigment. Look last slides Lecture 3"""
        n = X.shape[1]
        h,p = self.predict(X,loss_function)
        if loss_function == 'cross-entropy':
            g_batch = -(labels_onehot-p)

            j_wtr_w2 = 1/n*np.dot(g_batch,h.T) + 2*lambda_reg*self.layers[1].W
            j_wrt_b2 = 1/n*np.dot(g_batch,np.ones((n,1)))

            g_batch = np.dot(self.layers[1].W.T, g_batch)
            g_batch = np.multiply(g_batch,np.heaviside(h,0))

            j_wrt_b1 = 1/n*np.dot(g_batch,np.ones((n,1)))
            j_wtr_w1 = 1/n*np.dot(g_batch,X.T) + 2*lambda_reg*self.layers[0].W

            return j_wtr_w1,j_wrt_b1,j_wtr_w2,j_wrt_b2


    def fit(self,X_train,Y_train,X_val,Y_val,Ylabels,loss_function,n_batch,eta,n_epochs,lamda,eta_min,eta_max,stepsize,cycles):
            n = X_train.shape[1]
            error_train =[]
            error_val = []
            t = 0
            n_epochs = int(stepsize*cycles*2/n_batch)
            print("EPOCHS")
            print(n_epochs)
            for l in range(cycles):
                for i in tqdm(range(n_epochs)):
                    indices = np.arange(Y_train.shape[1])
                    np.random.shuffle(indices)
                    X_train = X_train[:,indices]
                    Y_train = Y_train[:,indices]
                    for j in range(int(n/n_batch)):
                        j = j + 1
                        j_start = (j - 1) * n_batch + 1
                        j_end = j*n_batch
                        X_batch = X_train[:, j_start:j_end]
                        Y_batch = Y_train[:,j_start:j_end]
                        j_wtr_w1,j_wtr_b1,j_wtr_w2,j_wtr_b2 = self.compute_gradients(X_batch,Y_batch,
                                                                                     loss_function,lamda)
                        self.layers[0].W = self.layers[0].W  - eta*j_wtr_w1
                        self.layers[0].b = self.layers[0].b - eta * j_wtr_b1
                        self.layers[1].W = self.layers[1].W  - eta*j_wtr_w2
                        self.layers[1].b = self.layers[1].b - eta * j_wtr_b2
                        t += 1
                        eta = self.set_eta(eta_min,eta_max,stepsize,l,t)
                        print(t)

                    _,prediction_train = self.predict(X_train,loss_function)
                    cost_train = self.compute_cost(X_train, Y_train, prediction_train,loss_function,lamda)
                    print("Cost train calculated!")
                    _,prediction_val = self.predict(X_val, loss_function)
                    cost_val = self.compute_cost(X_val, Y_val,prediction_val, loss_function,lamda)
                    print("Cost validation calculated!")
                    error_train.append(cost_train)
                    error_val.append(cost_val)
                    print("Step_moment #" + str(t) + " Training error:" + str(cost_train))
                    print("Step_moment #" + str(t) + " Training error:" + str(cost_val))
                    accuracy = self.compute_accuracy(Ylabels,prediction_train)
                    print("Train accuracy", accuracy)


            return {'loss_train':error_train,
                    'loss_val':error_val}

    def set_eta(self,eta_min,eta_max,ns,l,t):
        if 2*l*ns<t<(2*l+1)*ns:
            return eta_min + (t-2*l*ns)/ns*(eta_max-eta_min)
        if (2*l+1)*ns <= t <= 2*(l+1)*ns:
            return eta_max - (t-2*l*ns)/ns*(eta_max-eta_min)

    def ComputeCost(self, X, Y, W1,W2,b1, b2, lamda):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        _, prediction = self.Predict(X,W1,W2,b1,b2)
        entr = self._loss(Y, prediction,'cross-entropy')
        return None,1/num_datapoints*np.sum(entr)+ lamda*(np.sum(np.square(W1)) +
                                                                 np.sum(np.square(W2)))

    def Predict_layer(self,X,W,b,activation_function):
        if activation_function == 'softmax':
            return softmax(np.dot(W,X)+b)
        if activation_function == 'relu':
            return np.maximum(np.zeros(shape=(np.dot(W, X) + b).shape), np.dot(W, X) + b)


    def Predict(self,X,W1,W2,b1,b2):
        h = self.Predict_layer(X,W1, b1, 'relu')
        prediction = self.Predict_layer(h, W2, b2, 'softmax')
        return h,prediction

    def ComputeGradsNum(self,X, Y, P, W1, W2, b1, b2, lam, h):
        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)
        _, c = self.ComputeCost(X, Y, W1, W2, b1, b2, lam)
        for i in tqdm(range(len(b1))):
            b1_try = np.array(b1)
            b1_try[i] += h
            _, c2 = self.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)
            grad_b1[i] = (c2 - c) / h
        for i in tqdm(range(W1.shape[0])):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i, j] += h
                _, c2 = self.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)
                grad_W1[i, j] = (c2 - c) / h

        for i in tqdm(range(len(b2))):
            b2_try = np.array(b2)
            b2_try[i] += h
            _, c2 = self.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)
            grad_b2[i] = (c2 - c) / h

        for i in tqdm(range(W2.shape[0])):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i, j] += h
                _, c2 = self.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)
                grad_W2[i, j] = (c2 - c) / h

        return [grad_W1, grad_W2, grad_b1, grad_b2]


