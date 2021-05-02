import numpy as np
from src.utils import softmax
import math

class Layer:

    def __init__(self,n,input_nodes):
        np.random.seed(400)
        xavier = math.sqrt(6)/(math.sqrt(n+input_nodes))
        self.W = np.random.uniform(-xavier,xavier,size=(n, input_nodes))
        self.b = np.zeros(shape=(n, 1))
        self.X = np.array([[]])
        self.Sbatch = np.array([[]])
        self.Shat_batch = np.array([[]])
        self.Prediction = np.array([[]])
        self.mean = np.array([[]])
        self.variance = np.array([[]])
        self.gamma = np.ones(shape=(n, 1)) #ones
        self.beta = np.zeros(shape=(n, 1))#zeros
        self.mean_avg = np.array([[]])
        self.variance_avg = np.array([[]])

    def predict_layer(self,X,activation_function):
        self.X = X
        if activation_function == 'softmax':
            return softmax(np.dot(self.W,X)+self.b)
        if activation_function == 'relu':
            return np.maximum(np.zeros(shape=(np.dot(self.W, X) + self.b).shape), np.dot(self.W, X) + self.b)
        if activation_function == 'batch':
            self.Sbatch = np.dot(self.W, X) + self.b
            return self.Sbatch

    def backward_layer(self,Xbatch,Gbatch,eta,lambda_reg,batch_norm=False):
        n = Xbatch.shape[1]
        if batch_norm==False:
            j_wrt_w = 1 / n * np.dot(Gbatch, Xbatch.T) + 2 * lambda_reg * self.W
            j_wrt_b = 1 / n * np.dot(Gbatch, np.ones((n, 1)))
            Gbatch = np.dot(self.W.T, Gbatch)
            Gbatch = np.multiply(Gbatch, np.heaviside(Xbatch, 0))
            self.update_weights(j_wrt_w,j_wrt_b,eta,0,0)
            return Gbatch,j_wrt_w,j_wrt_b
        else:
            j_wrt_gamma = 1/n* np.dot(np.multiply(Gbatch, self.Shat_batch),np.ones((n, 1)))
            j_wrt_beta = 1 / n * np.dot(Gbatch, np.ones((n, 1)))
            Gbatch = np.multiply(Gbatch,np.dot(self.gamma,np.ones((n, 1)).T))
            Gbatch = self.batch_normbackpass(Gbatch)
            j_wrt_w = 1 / n * np.dot(Gbatch, Xbatch.T) + 2 * lambda_reg * self.W
            j_wrt_b = 1 / n * np.dot(Gbatch, np.ones((n, 1)))
            Gbatch = np.dot(self.W.T,Gbatch)
            Gbatch = np.multiply(Gbatch,np.heaviside(Xbatch, 0))
            self.update_weights(j_wrt_w,j_wrt_b,eta,j_wrt_gamma,j_wrt_beta)
            return Gbatch,j_wrt_w,j_wrt_b,j_wrt_gamma,j_wrt_beta

    def batch_normbackpass(self,gbatch):
        n = gbatch.shape[1]
        sigma_1 = (self.variance[:,np.newaxis] + np.finfo(float).eps)**(-0.5)
        sigma_2 = (self.variance[:,np.newaxis] + np.finfo(float).eps)**(-1.5)
        g1 = np.multiply(gbatch ,sigma_1@np.ones((n, 1)).T)
        g2 = np.multiply(gbatch, np.dot(sigma_2,np.ones((n, 1)).T))
        d = self.Sbatch - np.dot(self.mean,np.ones((n, 1)).T)
        c = np.dot(np.multiply(g2,d),np.ones((n, 1)))
        gbatch = g1 - 1/n*((g1@np.ones((n, 1)))@np.ones((n, 1)).T) - 1/n*np.multiply(d,np.dot(c,np.ones((n, 1)).T))
        return gbatch

    def update_weights(self,j_wrt_w,j_wrt_b,eta,j_wrt_gamma,j_wrt_beta):
        self.W = self.W - eta * j_wrt_w
        self.b = self.b - eta * j_wrt_b
        self.gamma = self.gamma - eta*j_wrt_gamma
        self.beta = self.beta -eta*j_wrt_beta



    def compute_mean_variance(self, s):
        alpha = 0.9
        n = s.shape[1]
        self.mean = np.mean(s,axis=1).reshape(-1, 1)
        self.variance = 1/n*np.sum((s-self.mean)**2,axis=1)
        if(self.mean_avg.shape[1]==0):
            self.mean_avg = self.mean
            self.variance_avg = self.variance
        self.mean_avg = alpha*self.mean_avg + (1-alpha)*self.mean
        self.variance_avg = alpha*self.variance_avg + (1-alpha)*self.variance