import numpy as np
from src.utils import softmax
from data.layer import Layer
from tqdm import tqdm
import random

class Classifier:
    def __init__(self):
        self.layers = []

    def add_layer(self, n, input_nodes,sigma_exp=None):
        layer = Layer(n, input_nodes,sigma_exp)
        self.layers.append(layer)

    def predict(self, X, complete=False,batch_normalization=False,training=False,dropout=False,p=0.7):
        predictions = [X]
        for i in range(len(self.layers) - 1):
            if batch_normalization:
                s = self.layers[i].predict_layer(X, 'batch')
                self.layers[i].compute_mean_variance(s)
                if dropout:
                    p = 0.9
                    u1 = np.random.binomial(1, p, size=s.shape) / p
                    s *= u1
                if training:
                    mean = self.layers[i].mean
                    variance = self.layers[i].variance
                else:
                    mean = self.layers[i].mean_avg
                    variance = self.layers[i].variance_avg
                s_tild = self.batch_normalization(s,mean,variance)
                self.layers[i].Shat_batch = s_tild
                s_f = np.multiply(self.layers[i].gamma,s_tild) + self.layers[i].beta
                X = np.maximum(np.zeros(shape=(s_f.shape)),s_f)
            else:
                X = self.layers[i].predict_layer(X, 'relu')
            predictions.append(X)
        prediction = self.layers[-1].predict_layer(X, 'softmax')
        self.layers[-1].Prediction = prediction
        predictions.append(prediction)
        if complete:
            return predictions
        else:
            return prediction

    """def predict(self, X, complete=False):
        predictions = [X]
        for i in range(len(self.layers) - 1):
            X = self.layers[i].predict_layer(X, 'relu')
            predictions.append(X)
        prediction = self.layers[-1].predict_layer(X, 'softmax')
        predictions.append(prediction)
        if complete:
            return predictions
        else:
            return prediction"""

    def batch_normalization(self,s,mu,v):
        return np.linalg.inv(np.diag(np.sqrt(v + np.finfo(float).eps))) @ (s - mu)


    def compute_cost(self, X, labels_onehot, prediction, loss_function, lambda_reg):
        """Equation number (5) in the paper"""
        num_datapoints = X.shape[1]
        entr = self._loss(labels_onehot, prediction, loss_function)
        return 1 / num_datapoints * np.sum(entr) + lambda_reg * (
                    np.sum(self.layers[0].W ** 2) + np.sum(self.layers[1].W))

    def _loss(self, label_onehotencoded, probabilities, loss_function):
        """Equation number (6) in the assigment"""
        if loss_function == 'cross-entropy':
            return -np.log(np.sum(label_onehotencoded * probabilities, axis=0))
        if loss_function == 'svm':
            s_y = np.sum(label_onehotencoded * probabilities, axis=0)
            s = np.maximum(np.zeros(probabilities.shape), probabilities - s_y + 1)
            s[label_onehotencoded.astype(bool)] = 0
            return np.sum(s, axis=0)

    def compute_accuracy(self, labels, prediction):
        """Equation number (4) in the assigment"""
        pred = np.argmax(prediction, axis=0)
        return np.sum(pred == labels) / len(prediction[0]) * 100

    def compute_gradients(self, X, labels_onehot, lambda_reg, eta, batch_norm=False,dropout=False):
        """Equation (10) and (11) in the assigment. Look last slides Lecture 3"""
        if batch_norm == False:
            n = X.shape[1]
            predictions_layers = self.predict(X, complete=True)
            g_batch = -(labels_onehot - predictions_layers[-1])
            weights = []
            bias = []
            for l in range(len(self.layers) - 1, 0, -1):
                g_batch, j_wtr_w2, j_wrt_b2 = self.layers[l].backward_layer(predictions_layers[l], g_batch, eta, lambda_reg)
                weights.append(j_wtr_w2)
                bias.append(j_wrt_b2)

            #first layer
            j_wrt_b1 = 1 / n * np.dot(g_batch, np.ones((n, 1)))
            j_wrt_w1 = 1 / n * np.dot(g_batch, X.T) + 2 * lambda_reg * self.layers[0].W
            self.layers[0].update_weights(j_wrt_w1,j_wrt_b1,eta,0,0)

            weights.append(j_wrt_w1)
            bias.append(j_wrt_b1)
            weights.reverse()
            bias.reverse()
            return weights, bias
        else:
            n = X.shape[1]
            predictions_layers = self.predict(X, complete=True,batch_normalization=True,training=True,dropout=dropout)
            g_batch = -(labels_onehot - predictions_layers[-1])
            weights = []
            bias = []
            gammas = []
            betas = []
            # last layer
            g_batch, j_wrt_w2, j_wrt_b2  = self.layers[-1].backward_layer(self.layers[-1].X,g_batch,eta,lambda_reg)
            self.layers[-1].update_weights(j_wrt_w2,j_wrt_b2,eta,0,0)
            weights.append(j_wrt_w2)
            bias.append(j_wrt_b2)
            # other layers
            for l in range(len(self.layers) - 2, -1, -1):
                g_batch,j_wrt_w,j_wrt_b,j_wrt_gamma,j_wrt_beta =self.layers[l].backward_layer(self.layers[l].X,
                                                                                            g_batch, eta,
                                                                             lambda_reg,batch_norm=True)
                weights.append(j_wrt_w)
                bias.append(j_wrt_b)
                gammas.append(j_wrt_gamma)
                betas.append(j_wrt_beta)
            weights.reverse()
            bias.reverse()
            gammas.reverse()
            betas.reverse()

            return weights,bias,gammas,betas

    def fit(self, X_train, Y_train, X_val, Y_val, Ylabelstrain, Ylabelsval, loss_function, n_batch, eta,
            n_epochs, lamda, eta_min, eta_max, n_s,batch_norm=False,dropout=False):

        cost_train_total = []
        cost_val_total = []
        loss_train_total = []
        loss_val_total = []
        acc_train_total = []
        acc_val_total = []
        n = X_train.shape[1]
        t = 0
        # 490 is one epoch
        for i in tqdm(range(n_epochs)):
            indices = np.arange(Y_train.shape[1])
            np.random.shuffle(indices)
            X_shuffled = X_train[:,indices]
            Y_shuffled = Y_train[:,indices]
            for j in range(int(n / n_batch)):
                j = j + 1
                j_start = (j - 1) * n_batch + 1
                j_end = j * n_batch
                X_batch = X_shuffled[:, j_start:j_end]
                Y_batch = Y_shuffled[:, j_start:j_end]

                # Update weights
                self.compute_gradients(X_batch, Y_batch, lamda, eta,batch_norm,dropout)
                t = (t + 1) % (2 * n_s)
                eta = self.set_eta(eta_min, eta_max, n_s, t)

            # Predict X_train and X_val
            prediction_train = self.predict(X_train,batch_normalization=batch_norm)
            prediction_val = self.predict(X_val,batch_normalization=batch_norm)

            """# Compute cost
            cost_train = self.compute_cost(X_train, Y_train, prediction_train, loss_function, lamda)
            cost_val = self.compute_cost(X_val, Y_val, prediction_val, loss_function, lamda)
            cost_train_total.append(cost_train)
            cost_val_total.append(cost_val)
            print("Step_moment #" + str(t) + " Training error:" + str(cost_train))
            print("Step_moment #" + str(t) + " Validation error:" + str(cost_val))"""

            # Compute accuracy
            train_accuracy = self.compute_accuracy(Ylabelstrain, prediction_train)
            accuracy_val = self.compute_accuracy(Ylabelsval, prediction_val)
            acc_train_total.append(train_accuracy)
            acc_val_total.append(accuracy_val)
            # print("Train accuracy", train_accuracy)
            # print("Validation accuracy", accuracy_val)

            # Compute loss
            entr = self._loss(Y_train, prediction_train, loss_function)
            loss_train = 1 / n * np.sum(entr)
            loss_train_total.append(loss_train)

            num_datapoints = X_val.shape[1]
            entr = self._loss(Y_val, prediction_val, loss_function)
            loss_val = 1 / num_datapoints * np.sum(entr)
            loss_val_total.append(loss_val)

        return {'cost_train': cost_train_total,
                'cost_val': cost_val_total,
                'accuracy_train': acc_train_total,
                'accuracy_val': acc_val_total,
                'loss_train': loss_train_total,
                'loss_val': loss_val_total}


    def set_eta(self, eta_min, eta_max, ns, t):
        if 0 <= t < ns:
            return eta_min + (t) / ns * (eta_max - eta_min)
        if ns <= t <= 2 * ns:
            return eta_max - (t - ns) / ns * (eta_max - eta_min)

    def compute_cost_numerically(self, X, Y, lamda,batch=False):
        if batch == False:
            num_datapoints = X.shape[1]
            predictions = self.predict(X, complete=True)
            entr = self._loss(Y, predictions[-1], 'cross-entropy')
            return None, 1 / num_datapoints * np.sum(entr)
        else:
            num_datapoints = X.shape[1]
            predictions = self.predict(X, complete=True,batch_normalization=True,training=True)
            entr = self._loss(Y, predictions[-1], 'cross-entropy')
            return 1 / num_datapoints * np.sum(entr)


    def check_gradients_num(self, x, y, h):
        W_grads = []
        b_grads = []
        for layer in tqdm(self.layers):
            # Create grads for layer
            W_grad = np.zeros_like(layer.W)
            b_grad = np.zeros_like(layer.b)
            #  Check gradient for W  and b.
            for i in range(len(layer.W)):
                for j in range(len(layer.W[i])):
                    layer.W[i][j] -= h
                    _, c1 = self.compute_cost_numerically(x, y, 0)
                    layer.W[i][j] += 2 * h
                    _, c2 = self.compute_cost_numerically(x, y, 0)
                    W_grad[i][j] = (c2 - c1) / (2 * h)
                    layer.W[i][j] -= h
            for i in range(len(layer.b)):
                layer.b[i] -= h
                _, c1 = self.compute_cost_numerically(x, y, 0)
                layer.b[i] += 2 * h
                _, c2 = self.compute_cost_numerically(x, y, 0)
                b_grad[i] = (c2 - c1) / (2 * h)
                layer.b[i] -= h
            W_grads.append(W_grad)
            b_grads.append(b_grad)
        return W_grads, b_grads



    def check_gradients_num_batch(self, x, y, h):
        grad_W = []
        grad_b = []
        grad_gamma = []
        grad_beta = []

        for i in range(len(self.layers)):
            print('Computing numerical gradients... Layer ', str(i + 1))
            grad_W.append(np.zeros_like(self.layers[i].W))
            grad_b.append(np.zeros_like(self.layers[i].b))

            for j in range(len(self.layers[i].b)):
                self.layers[i].b[j] -= h
                c1 = self.compute_cost_numerically(x, y,0,True)
                self.layers[i].b[j] += 2 * h
                c2 = self.compute_cost_numerically(x, y,0,True)

                grad_b[i][j] = (c2 - c1) / (2 * h)
                self.layers[i].b[j] -= h

            for j in range(self.layers[i].W.shape[0]):
                for l in range(self.layers[i].W.shape[1]):
                    self.layers[i].W[j, l] -= h
                    c1 = self.compute_cost_numerically(x, y,0,True)
                    self.layers[i].W[j, l] += 2 * h
                    c2 = self.compute_cost_numerically(x, y,0,True)

                    grad_W[i][j, l] = (c2 - c1) / (2 * h)
                    self.layers[i].W[j, l] -= h

        for i in range(len(self.layers) - 1):
            grad_gamma.append(np.zeros_like(self.layers[i].gamma))
            grad_beta.append(np.zeros_like(self.layers[i].beta))

            for j in range(len(self.layers[i].gamma)):
                self.layers[i].gamma[j] -= h
                c1 = self.compute_cost_numerically(x, y,0,True)
                self.layers[i].gamma[j] += 2 * h
                c2 = self.compute_cost_numerically(x, y,0,True)

                grad_gamma[i][j] = (c2 - c1) / (2 * h)
                self.layers[i].gamma[j] -= h

            for j in range(len(self.layers[i].beta)):
                self.layers[i].beta[j] -= h
                c1 = self.compute_cost_numerically(x, y,0,True)
                self.layers[i].beta[j] += 2 * h
                c2 = self.compute_cost_numerically(x, y,0,True)

                grad_beta[i][j] = (c2 - c1) / (2 * h)
                self.layers[i].beta[j] -= h

        return grad_W, grad_b, grad_gamma, grad_beta