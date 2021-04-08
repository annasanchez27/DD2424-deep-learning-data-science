import numpy as np


class Layer:

    def __init__(self,n,input_nodes):
        self.W = np.random.normal(0, 0.01, size=(n, input_nodes))
        self.b = np.zeros(shape=(n, 1))