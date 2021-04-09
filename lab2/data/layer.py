import numpy as np


class Layer:

    def __init__(self,n,input_nodes):
        np.random.seed(400)
        self.W = np.random.normal(0,1/input_nodes, size=(n, input_nodes))
        self.b = np.zeros(shape=(n, 1))