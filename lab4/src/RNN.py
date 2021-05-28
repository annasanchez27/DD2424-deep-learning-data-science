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
        h = h.reshape(-1,1)
        x = x.reshape(-1,1)
        a_t = np.dot(self.W, h) + np.dot(self.U,x) + self.b
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
            x0 = np.zeros((self.k, 1))
            x0[idx] = 1
            sequence += self.ind_to_char[idx]
        return sequence

    def compute_gradients(self, read_data, sequence_length):
        # Sequence of input characters from the text
        input_chars = read_data['data'][0: sequence_length]
        target_chars = read_data['data'][1: sequence_length + 1]

        # Convert to one-hot encoding
        input_labels = self.encode(input_chars, read_data['char_to_ind'])
        target_labels = self.encode(target_chars, read_data['char_to_ind'])
        print(input_labels.shape)

        # Initialize storage
        p_list, h_list = [], []

        # Initial state
        h0 = np.zeros(self.m)
        h_list.append(h0)

        # Forward pass
        for t in range(sequence_length):
            h, p = self.forward_pass(h_list[-1], input_labels[:, t])
            h_list.append(h)
            p_list.append(p)

        # Loss
        p_list = np.array(p_list)[:,:,0].T
        loss = -sum(np.log(np.multiply(target_labels, p_list).sum(axis=0)))
        print(f"The loss is {loss}")

    def encode(self,input_text, char_to_ind):
        indices = [char_to_ind[char] for char in input_text]
        one_hot_encoding = (np.eye(len(char_to_ind.keys()))[indices]).T
        return one_hot_encoding