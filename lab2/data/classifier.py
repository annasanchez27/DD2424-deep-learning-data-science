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

    def predict_dropout(self,X,loss_function,p=0.7):
        h = self.predict_layer(X, self.layers[0].W, self.layers[0].b, 'relu')
        u1 = np.random.binomial(1, p, size=h.shape) / p
        h *= u1
        prediction = self.predict_layer(h, self.layers[1].W, self.layers[1].b, 'softmax')
        u2 = np.random.binomial(1, p, size=prediction.shape) / p
        prediction *= u2
        return h, prediction

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

    def compute_gradients(self,X,labels_onehot,loss_function,lambda_reg,dropout,jitter):
        """Equation (10) and (11) in the assigment. Look last slides Lecture 3"""
        n = X.shape[1]
        if jitter:
            noise = np.random.normal(0, 0.1, size=X.shape)
            X = X + noise
        if dropout:
            h,p = self.predict_dropout(X,loss_function)
        else:
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


    def fit(self,X_train,Y_train,X_val,Y_val,Ylabelstrain,Ylabelsval,loss_function,n_batch,eta,
            n_epochs,lamda,eta_min,eta_max,n_s,dropout=False,jitter=False):

            cost_train_total =[]
            cost_val_total = []
            loss_train_total = []
            loss_val_total = []
            acc_train_total = []
            acc_val_total = []
            etas = []

            n = X_train.shape[1]
            t = 0
            #490 is one epoch
            for i in tqdm(range(n_epochs)):
                for j in range(int(n/n_batch)):
                    print(j)
                    #Select the batch
                    j = j + 1
                    j_start = (j - 1) * n_batch + 1
                    j_end = j*n_batch
                    X_batch = X_train[:, j_start:j_end]
                    Y_batch = Y_train[:,j_start:j_end]

                    #Update weights
                    self.update_weights(X_batch,Y_batch,eta,lamda,loss_function,dropout,jitter)
                    t = (t+1) % (2*n_s)

                    eta = self.set_eta_test(eta_min,eta_max,n_s,t)

                #Predict X_train and X_val
                _,prediction_train = self.predict(X_train,loss_function)
                _, prediction_val = self.predict(X_val, loss_function)

                #Compute cost
                print("ETA MAX", eta_max)
                print(eta)
                etas.append(eta)
                cost_train = self.compute_cost(X_train, Y_train, prediction_train,loss_function,lamda)
                cost_val = self.compute_cost(X_val, Y_val,prediction_val, loss_function,lamda)
                cost_train_total.append(cost_train)
                cost_val_total.append(cost_val)
                print("Step_moment #" + str(t) + " Training error:" + str(cost_train))
                print("Step_moment #" + str(t) + " Validation error:" + str(cost_val))

                #Compute accuracy
                train_accuracy = self.compute_accuracy(Ylabelstrain,prediction_train)
                accuracy_val = self.compute_accuracy(Ylabelsval, prediction_val)
                acc_train_total.append(train_accuracy)
                acc_val_total.append(accuracy_val)
                #print("Train accuracy", train_accuracy)
                #print("Validation accuracy", accuracy_val)

                #Compute loss
                num_datapoints = X_train.shape[1]
                entr = self._loss(Y_train, prediction_train, loss_function)
                loss_train = 1 / num_datapoints * np.sum(entr)
                loss_train_total.append(loss_train)

                num_datapoints = X_val.shape[1]
                entr = self._loss(Y_val, prediction_val, loss_function)
                loss_val = 1 / num_datapoints * np.sum(entr)
                loss_val_total.append(loss_val)
                print("Step_moment #" + str(t) + " Training loss:" + str(loss_train))
                print("Step_moment #" + str(t) + " Validation loss:" + str(loss_val))

            return {'cost_train':cost_train_total,
                    'cost_val':cost_val_total,
                    'accuracy_train':acc_train_total,
                    'accuracy_val':acc_val_total,
                    'loss_train': loss_train_total,
                    'loss_val': loss_val_total,
                    'eta': etas}







    def update_weights(self,X_batch, Y_batch,eta,lamda, loss_function,dropout,jitter):
        # Compute the gradients
        j_wtr_w1, j_wtr_b1, j_wtr_w2, j_wtr_b2 = self.compute_gradients(X_batch, Y_batch,
                                                                        loss_function, lamda,dropout,jitter)

        # Update the weights
        self.layers[0].W = self.layers[0].W - eta * j_wtr_w1
        self.layers[0].b = self.layers[0].b - eta * j_wtr_b1
        self.layers[1].W = self.layers[1].W - eta * j_wtr_w2
        self.layers[1].b = self.layers[1].b - eta * j_wtr_b2


    def set_eta(self,eta_min,eta_max,ns,t):
        if 0<=t<ns:
            return eta_min + (t)/ns*(eta_max-eta_min)
        if ns <= t <= 2*ns:
            if (t==ns):
                print("MAXIMUM REACHED!")
            return eta_max - (t-ns)/ns*(eta_max-eta_min)

    def set_eta_test(self,eta_min,eta_max,ns,t):
        return eta_min + t*(eta_max-eta_min)/(2*ns)

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


