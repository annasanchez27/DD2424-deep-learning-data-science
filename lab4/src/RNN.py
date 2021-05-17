import numpy as np
import random


class RNN:
    def __init__(self, k, m=100, sig=0.01):
        # TODO: set eta
        # TODO: set length of input sequence
        self.k = k
        self.m = m
        self.b = np.zeros((m, 1))
        self.c = np.zeros((k, 1))
        self.U = np.random.normal(0, sig, size=(m, k))
        self.W = np.random.normal(0, sig, size=(m, m))
        self.V = np.random.normal(0, sig, size=(k, m))
        self.ind_to_char = None
        self.char_to_ind = None

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward_pass(self, h, x):
        a_t = self.W @ h + self.U @ x + self.b
        h = np.tanh(a_t)
        o_t = self.V @ h + self.c
        p_t = self.softmax(o_t)
        return h, p_t

    def synthesize_seq_chars(self, h0, x0, n):
        """
        Synthesize a sequence of characters using
        the current parameter values in the RNN.
        (Eq 1- Eq 4)
        :param h0: hidden state at time 0
        :param x0: first (dummy) input vector to your RNN
        :param n: length of the sequence that will be generated
        :return:
        """
        sequence = ""
        for t in range(n):
            h, p_t = self.forward_pass(h0, x0)
            # Randomly select character
            cum_sum = np.cumsum(p_t)
            a = random.uniform(0, 1)
            idx = np.where(np.logical_and(cum_sum >= a))[0]
            xnext = np.zeros((self.k, 1))
            xnext[idx] = 1
            sequence += self.ind_to_char[idx]
        return sequence
