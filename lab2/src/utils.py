import numpy as np
import matplotlib.pyplot as plt

def one_hot_encoder(vector):
    n_values = np.max(vector) + 1
    b = np.eye(n_values)[vector]
    return b.T

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()

def error_plot(train_error,validation_error,ylabel):
    plt.plot([100*(i+1) for i in range(10)], train_error,label="training")
    plt.plot([100*(i+1) for i in range(10)],validation_error,label="validation")
    plt.legend()
    plt.xlabel("update step")
    plt.ylabel(ylabel)
    plt.show()

def error_plot3(train_error,validation_error,ylabel):
    plt.plot([100*(i+1) for i in range(50)], train_error,label="training")
    plt.plot([100*(i+1) for i in range(50)],validation_error,label="validation")
    plt.legend()
    plt.xlabel("update step")
    plt.ylabel(ylabel)
    plt.show()