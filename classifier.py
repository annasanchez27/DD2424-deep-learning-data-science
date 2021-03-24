import numpy as np
from utils import softmax

class Classifier:

    def __init__(self,dim_images,num_labels):
        self.W = np.random.normal(0, 0.01, size=(num_labels, dim_images))
        self.b = np.random.normal(0, 0.01, size=(num_labels, 1))

    def predict(self, X):
        return softmax(np.dot(self.W,X)+self.b)

    def compute_cost(self,data,prediction,lambda_reg=1.0):
        """Equation number (5) in the paper"""
        num_datapoints = data['data'].shape[1]
        entr = self._cross_entropy(data['one_hot'], prediction)
        return 1/num_datapoints*np.sum(entr)+ lambda_reg*np.sum(np.square(self.W))


    def _cross_entropy(self,label_onehotencoded,probabilities):
        """Equation number (6) in the assigment"""
        r = -np.log(np.sum(label_onehotencoded * probabilities,axis=0))
        return r

    def compute_accuracy(self,data,prediction):
        """Equation number (4) in the assigment"""
        pred = np.argmax(prediction,axis=0)
        return np.sum(pred == data['labels'])/len(prediction[0])*100
