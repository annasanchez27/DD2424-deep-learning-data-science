import numpy as np
from lab3.src.utils import softmax

class Layer:

    def __init__(self,n,input_nodes):
        np.random.seed(400)
        self.W = np.random.normal(0,1/input_nodes, size=(n, input_nodes))
        self.b = np.zeros(shape=(n, 1))

    def predict_layer(self,X,W,b,activation_function):
        if activation_function == 'softmax':
            return softmax(np.dot(W,X)+b)
        if activation_function == 'relu':
            return np.maximum(np.zeros(shape=(np.dot(W, X) + b).shape), np.dot(W, X) + b)