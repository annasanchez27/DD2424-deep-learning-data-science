import copy
import numpy as np

def one_hot(character_index, number_distinct_characters):
    character_one_hot = np.zeros(shape=(number_distinct_characters, 1))
    character_one_hot[character_index, 0] = 1

    return character_one_hot


def encode(input_text, char_to_ind):
    indices = [char_to_ind[char] for char in input_text]
    one_hot_encoding = (np.eye(len(char_to_ind.keys()))[indices]).T
    return one_hot_encoding


class RNN(object):

    def __init__(self, k, m=100, sig=0.01):
        np.random.seed(42)
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

    def calculate_loss(self, target_labels, p):
        loss = 0
        for t in range(len(p)):
            loss -= np.log(np.dot(target_labels[:, [t]].T, p[t]))[0, 0]
        return loss

    def blank_parameters(self):
        return {'U': np.zeros(self.param['U'].shape), 'V': np.zeros(self.param['V'].shape),
                'W': np.zeros(self.param['W'].shape), 'b': np.zeros(self.param['b'].shape),
                'c': np.zeros(self.param['c'].shape)}

    def backward(self, input_labels, target_labels, p_list, h_list, a_list):
        input_labels = np.array(input_labels)
        target_labels = np.array(target_labels)
        p_list = np.array(p_list)[:, :, 0].T
        h_list = np.array(h_list)[:, :, 0].T
        sequence_length = input_labels.shape[1]
        h = h_list[:, 1:]
        l_wrt_h = np.zeros((self.m, sequence_length))
        l_wrt_a = np.zeros((self.m, sequence_length))
        grads = self.blank_parameters()

        # Backward pass
        l_wrt_o = -(target_labels - p_list).T  # (5,80)
        grads['V'] = np.dot(l_wrt_o.T, h.T)  # (80,100) = (80,5) (5,100)
        grads['c'] = np.sum(l_wrt_o.T, axis=1).reshape(-1, 1)  # (80,1)

        l_wrt_h[:, -1] = np.dot(l_wrt_o.T[:, -1], self.param['V'])
        l_wrt_a[:, -1] = np.dot(l_wrt_h[:, -1], np.diag(1 - np.square(h_list[:, -1])))

        for t in reversed(range(sequence_length - 1)):
            l_wrt_h[:, t] = np.dot(l_wrt_o[t, :], self.param['V']) + \
                            np.dot(l_wrt_a[:, t + 1], self.param['W'])
            l_wrt_a[:, t] = np.dot(l_wrt_h[:, t], np.diag(1 - np.square(h_list[:, t + 1])))

        grads['U'] = np.dot(l_wrt_a, input_labels.T)
        grads['W'] = np.dot(l_wrt_a, h_list[:, :-1].T)
        grads['b'] = np.sum(l_wrt_a, axis=1).reshape(-1, 1)

        for grad in grads.keys():
            grads[grad] = np.clip(grads[grad], -5, 5)

        return grads

    def grads_numeric(self, X, Y, h=1e-4):
        h0 = np.zeros(shape=(self.m, 1))
        grads = self.blank_parameters()
        for parameter in self.param.keys():
            for i in range(self.param[parameter].shape[0]):
                for j in range(self.param[parameter].shape[1]):
                    copy_model = copy.deepcopy(self)
                    copy_model.param[parameter][i, j] += h
                    p, _, _ = copy_model.forward(X, h0)
                    loss2 = copy_model.calculate_loss(Y, p)
                    copy_model.param[parameter][i, j] -= 2 * h
                    p, _, _ = copy_model.forward(X, h0)
                    loss1 = copy_model.calculate_loss(Y, p)
                    grads[parameter][i, j] = (loss2 - loss1) / (2 * h)

        return grads


    def compute_gradients(self,X,Y,h0):
        p, h, a = self.forward(X, h0)
        loss = self.calculate_loss(Y, p)
        grads = self.backward(X, Y, p, h, a)
        return grads,loss,h[-1]


    def synthesize(self, x0, h0, length):
        x = x0
        h_prev = h0
        sequence = ""
        for t in range(length):
            a = self.param["W"]@ h_prev + self.param["U"] @ x + self.param["b"]
            h = np.tanh(a)
            o = self.param["V"] @ h + self.param["c"]
            p = self.softmax(o)
            x = np.zeros((self.k, 1))
            it = np.random.choice(range(self.k), p=p.flat)
            x[it] = 1
            sequence += self.ind_to_char[it]
            h_prev = h
        return sequence

    def fit(self,read_data,sequence_length,eta=0.1,max_epochs=10):
        e, iteration, epoch = 0, 0, 0
        loss_list = []
        data = read_data['data']

        rnn_params,mem_params = self.init_adagrad()

        while epoch < max_epochs:
            if iteration == 0 or e >= (len(data) - sequence_length - 1):
                e = 0
                h_prev = np.zeros((self.m, 1))
                epoch += 1

            input_chars = data[e: e + sequence_length]
            target_chars = data[e + 1: e + 1 + sequence_length]
            inputs = encode(input_chars, read_data['char_to_ind'])
            targets = encode(target_chars, read_data['char_to_ind'])
            grads, loss, h_prev = self.compute_gradients(inputs, targets,h_prev)

            if iteration == 0 and epoch == 1:
                smooth_loss = loss

            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            loss_list.append(smooth_loss)
            if iteration % 10 == 0:
                print("Loss", loss)
                print("Iteration", iteration)
                synthesize_one_hot = self.synthesize(inputs[:, [0]], h_prev, 1000)
                print(synthesize_one_hot)

            # Adagrad
            for key in rnn_params:
                mem_params[key] += grads[key] * grads[key]
                rnn_params[key] -= eta / np.sqrt(mem_params[key] +
                                                     np.finfo(float).eps) * grads[key]

            e += sequence_length
            iteration += 1
        print("LAST ONE",iteration, smooth_loss,epoch )
        synthesize_one_hot = self.synthesize(inputs[:, [0]], h_prev, 1000)
        print(synthesize_one_hot)
        return loss_list

    def init_adagrad(self):
        rnn = {"W": self.param["W"], "U": self.param["U"], "V": self.param["V"], "b": self.param["b"],
                      "c": self.param["c"]}

        mem = {"W": np.zeros(self.param["W"].shape), "U": np.zeros(self.param["U"].shape),
                      "V": np.zeros(self.param["V"].shape), "b": np.zeros(self.param["b"].shape),
                      "c": np.zeros(self.param["c"].shape)}

        return rnn,mem