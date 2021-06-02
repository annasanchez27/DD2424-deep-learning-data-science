def one_hot(character_index, number_distinct_characters):
    character_one_hot = np.zeros(shape=(number_distinct_characters, 1))
    character_one_hot[character_index, 0] = 1

    return character_one_hot


class RNN2(object):

    def __init__(self, k, m=5, sig=0.01, seed=42):
        self.k = k
        self.m = m
        self.ind_to_char = None
        self.char_to_ind = None
        self.param = {'U': np.random.normal(0, sig, size=(m, k)), 'V': np.random.normal(0, sig, size=(k, m)),
                      'W': np.random.normal(0, sig, size=(m, m)), 'b': np.zeros((m, 1)),
                      'c': np.zeros((k, 1))}

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(self, input_labels, h0):
        seq_length = input_labels.shape[1]
        p, h, a = [], [h0], []
        for t in range(seq_length):
            a.append(self.param['W'] @ h[-1] + self.param['U'] @ input_labels[:, [t]] + self.param['b'])
            h.append(np.tanh(a[t]))
            p.append(self.softmax(self.param['V'] @ h[-1] + self.param['c']))
        return p, h, a

    def calculate_loss(self, Y, p):
        loss = 0
        for t in range(len(p)):
            loss -= np.log(Y[:, [t]].T @ p[t])[0, 0]
        return loss

    def blank_parameters(self):
        return {'U': np.zeros(self.param['U'].shape), 'V': np.zeros(self.param['V'].shape),
                'W': np.zeros(self.param['W'].shape), 'b': np.zeros(self.param['b'].shape),
                'c': np.zeros(self.param['c'].shape)}

    def backward(self, X, Y, p, h, a):
        seq_length = X.shape[1]
        h0 = h[0]
        h = h[1:]
        grads = self.blank_parameters()
        grad_a = [None] * seq_length

        for t in range((seq_length - 1), -1, -1):
            g = -(Y[:, [t]] - p[t]).T
            grads['V'] += g.T @ h[t].T
            grads['c'] += g.T
            if t < (seq_length - 1):
                dL_h = g @ self.param['V'] + grad_a[t + 1] @ self.param['W']
            else:
                dL_h = g @ self.param['V']
            grad_a[t] = (dL_h @ np.diag(1 - np.square(h[t][:, 0])))
            if t == 0:
                grads['W'] += grad_a[t].T @ h0.T
            else:
                grads['W'] += grad_a[t].T @ h[t - 1].T
            grads['U'] += grad_a[t].T @ X[:, [t]].T
            grads['b'] += grad_a[t].T

        # Clipping gradients
        for parameter in ['b', 'c', 'U', 'W', 'V']:
            grads[parameter] = np.clip(grads[parameter], -5, 5)

        return grads   