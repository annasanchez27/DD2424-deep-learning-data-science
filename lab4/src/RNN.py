import numpy as np
import random
from tqdm import tqdm


def encode(input_text, char_to_ind):
    indices = [char_to_ind[char] for char in input_text]
    one_hot_encoding = (np.eye(len(char_to_ind.keys()))[indices]).T
    return one_hot_encoding


class RNN:
    def __init__(self, k, m=100, sig=0.001):
        random.seed(42)
        # TODO: set eta
        # TODO: set length of input sequence
        self.k = k
        self.m = m
        self.ind_to_char = None
        self.char_to_ind = None
        self.param = {'U': np.random.normal(0, sig, size=(m, k)), 'V': np.random.normal(0, sig, size=(k, m)),
                      'W': np.random.normal(0, sig, size=(m, m)), 'b': np.zeros((m, 1)),
                      'c': np.zeros((k, 1))}

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward_pass(self, h, x):
        h = h.reshape(-1, 1)
        x = x.reshape(-1, 1)
        a_t = np.dot(self.param['W'], h) + np.dot(self.param['U'], x) + self.param['b']
        h = np.tanh(a_t)
        o_t = self.param['V'] @ h + self.param['c']
        p_t = self.softmax(o_t)
        return h, p_t,a_t

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
        random.seed(42)
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

    def compute_gradients(self, input_labels, target_labels):
        sequence_length = input_labels.shape[1]
        # Initialize storage
        p_list, h_list, a_list = [], [], []

        # Initial state
        h0 = np.zeros((self.m, 1))
        h_list.append(h0)

        # Forward pass
        for t in range(sequence_length):
            h, p, a = self.forward_pass(h_list[-1], input_labels[:, t])
            h_list.append(h)
            p_list.append(p)
            a_list.append(a)
        # Loss
        p_list = np.array(p_list)[:, :, 0].T
        h_list = np.array(h_list)[:, :, 0].T
        a_list = np.array(a_list)[:, :, 0].T

        loss = self.calculate_loss(p_list, target_labels)

        # Backward pass
        grads = self.backward(input_labels, target_labels, p_list, h_list, sequence_length,a_list)

        return grads, loss

    def calculate_loss(self, p_list, target_labels):
        loss = -sum(np.log(np.multiply(target_labels, p_list).sum(axis=0)))
        return loss

    def blank_parameters(self):
        return {'U': np.zeros(self.param['U'].shape), 'V': np.zeros(self.param['V'].shape),
                'W': np.zeros(self.param['W'].shape), 'b': np.zeros(self.param['b'].shape),
                'c': np.zeros(self.param['c'].shape)}

    def backward2(self, x, y, p_list, H, sequence_length):
        H = H[:, 1:]

        grads = self.blank_parameters()
        grad_a = np.zeros(grads["b"].shape)
        for i in reversed(range(x.shape[1])):
            x_t = x[:, i].reshape(x.shape[0], 1)
            y_t = y[:, i].reshape(y.shape[0], 1)
            grad_o = -(y_t - p_list[:, i].reshape(-1, 1)).T  # dimension (1,80)

            g = grad_o  # dimension (1,80)
            grads["c"] += g.T  # dimension (80,1)

            grads["V"] = np.matmul(g.T, H[:, i].reshape(-1, 1).T)  # dimension 80,100

            grad_h = np.dot(self.param['V'].T, g.T) + np.dot(self.param['W'].T, grad_a)
            grad_a = grad_h * (1 - (H[:, i].reshape(-1, 1) ** 2))
            grads["b"] += grad_a  # dimension 100,1
            grads["W"] = np.matmul(g.reshape(-1, 1), H[:, i - 1].reshape(-1, 1).T)  # dimension (80,1) x (1,100)
            grads["U"] = np.matmul(g.reshape(-1, 1), x_t.T)
        return grads

    def backward(self, input_labels, target_labels, p_list, h_list, sequence_length,a_list):
        h = h_list[:, 1:]

        # Backward pass
        grads = self.blank_parameters()

        l_wrt_o = -(target_labels - p_list).T  # (5,80)
        grads['V'] = np.dot(l_wrt_o.T, h.T)  # (80,100) = (80,5) (5,100)
        grads['c'] = np.sum(l_wrt_o.T,axis=1).reshape(-1,1) # (80,1)

        l_wrt_h = np.zeros((self.m, sequence_length))
        l_wrt_a = np.zeros((self.m, sequence_length))

        # last hidden state
        l_wrt_h[:, -1] = np.dot(l_wrt_o.T[:, -1], self.param['V'])
        l_wrt_a[:, -1] = np.dot(l_wrt_h[:, -1], np.diag(1 - np.square(h_list[:, -1])))

        for t in reversed(range(sequence_length - 1)):
            l_wrt_h[:, t] = np.dot(l_wrt_o[t, :], self.param['V']) + \
                            np.dot(l_wrt_a[:, t + 1], self.param['W'])
            l_wrt_a[:, t] = np.dot(l_wrt_h[:, t], np.diag(1 - np.square(h_list[:, t + 1])))

        grads['U'] = np.dot(l_wrt_a, input_labels.T)
        grads['W'] = np.dot(l_wrt_a, h_list[:, :-1].T)
        grads['b'] = np.sum(l_wrt_a,axis=1).reshape(-1,1)
        grads['h'] = h[:-1]
        return grads

    def compute_grads_num(self, X, Y, h):
        grads = self.blank_parameters()
        for key in tqdm(self.param):
            for i in range(self.param[key].shape[0]):
                if self.param[key].ndim == 1:
                    self.param[key][i] -= h
                    _, l1 = self.compute_gradients(X, Y)
                    self.param[key][i] += 2 * h
                    _, l2 = self.compute_gradients(X, Y)
                    grads[key][i] = (l2 - l1) / (2 * h)
                    self.param[key][i] -= h
                else:
                    for j in range(self.param[key].shape[1]):
                        self.param[key][i, j] -= h
                        _, l1 = self.compute_gradients(X, Y)
                        self.param[key][i, j] += 2 * h
                        _, l2 = self.compute_gradients(X, Y)
                        grads[key][i, j] = (l2 - l1) / (2 * h)
                        self.param[key][i, j] -= h
        return grads


    def backward3(self,X, Y, P, H, sequence_length,a_list):
        grads = self.blank_parameters()
        H0 = H[:, :-1]
        H = H[:, 1:]
        G = -(Y.T - P.T).T

        grads['V'] = np.dot(G, H.T)
        grads['c'] = np.sum(G, axis=-1, keepdims=True)


        dLdh = np.zeros((X.shape[1], self.m))
        dLda = np.zeros((self.m, X.shape[1]))

        dLdh[-1] = np.dot(G.T[-1], self.param['V'])
        dLda[:,-1] = np.multiply(dLdh[-1].T, (1 - np.multiply(np.tanh(a_list[:, -1]), np.tanh(a_list[:, -1]))))

        for t in range(X.shape[1]-2, -1, -1):
            dLdh[t] = np.dot(G.T[t], self.param['V']) + np.dot(dLda[:, t+1],  self.param['W'])
            dLda[:,t] = np.multiply(dLdh[t].T, (1 - np.multiply(np.tanh(a_list[:, t]), np.tanh(a_list[:, t]))))

        grads['W'] = np.dot(dLda, H0.T)
        grads['U'] = np.dot(dLda, X.T)
        grads['b'] = np.sum(dLda, axis=-1, keepdims=True)
        return grads

    def fit(self, read_data,sequence_length, learning_rate=0.1):
        MAX_ITERATIONS = 100000
        data = read_data['data']
        e = 1
        smooth_loss = -np.log(1 / self.k) * sequence_length
        smooth_losses = []
        #h = np.zeros((self.m, 1))
        for it in range(MAX_ITERATIONS):
            if e >= len(data) - sequence_length - 1:
                e = 0
            input_chars = data[e: e + sequence_length]
            target_chars = data[e + 1: e + 1 + sequence_length]
            X = encode(input_chars, read_data['char_to_ind'])
            Y = encode(target_chars, read_data['char_to_ind'])

            grads,loss = self.compute_gradients(X, Y)
            #h = grads['h']
            smooth_loss = (0.999 * smooth_loss) + (0.001 * loss)
            smooth_losses.append(smooth_loss)
            for key in self.param.keys():
                self.param[key] += - (learning_rate * grads[key] ** 2) / np.sqrt(grads[key] ** 2 + 1e-8)

            if it % 10000 == 0:
                print("iter = " + str(it), "loss = " + str(smooth_loss))

            e += sequence_length