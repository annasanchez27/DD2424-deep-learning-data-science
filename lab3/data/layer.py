import numpy as np
from src.utils import softmax
import math

class Layer:

    def __init__(self,n,input_nodes):
        np.random.seed(400)
        xavier = math.sqrt(6)/(math.sqrt(n+input_nodes))
        self.W = np.random.uniform(-xavier,xavier,size=(n, input_nodes))
        self.b = np.zeros(shape=(n, 1))

    def predict_layer(self,X,activation_function):
        if activation_function == 'softmax':
            return softmax(np.dot(self.W,X)+self.b)
        if activation_function == 'relu':
            return np.maximum(np.zeros(shape=(np.dot(self.W, X) + self.b).shape), np.dot(self.W, X) + self.b)

    def backward_layer(self,Xbatch,Gbatch,eta,lambda_reg):
        n = Xbatch.shape[1]
        j_wrt_w = 1 / n * np.dot(Gbatch, Xbatch.T) + 2 * lambda_reg * self.W
        j_wrt_b = 1 / n * np.dot(Gbatch, np.ones((n, 1)))
        Gbatch = np.dot(self.W.T, Gbatch)
        Gbatch = np.multiply(Gbatch, np.heaviside(Xbatch, 0))
        self.update_weights(j_wrt_w,j_wrt_b,eta)
        return Gbatch,j_wrt_w,j_wrt_b

    def update_weights(self,j_wrt_w,j_wrt_b,eta):
        self.W = self.W - eta * j_wrt_w
        self.b = self.b - eta * j_wrt_b