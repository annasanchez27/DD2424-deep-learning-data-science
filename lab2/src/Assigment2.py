from lab2.data.load_data import load_data,preprocessing
from lab2.data.classifier import Classifier
import numpy as np
from utils import montage,error_plot







def main():
    data = load_data()
    data = preprocessing(data)

    d = len(data['train_data']['data'])
    m = len(data['train_data']['data'][0])

    k = len(data['train_data']['one_hot'])
    classifier = Classifier()
    classifier.add_layer(n=m, input_nodes=d)
    classifier.add_layer(n=k,input_nodes=m)
    loss = classifier.fit(data['train_data']['data'], data['train_data']['one_hot'],
                          data['validation_data']['data'], data['validation_data']['one_hot'],
                          loss_function='cross-entropy',
                          n_batch=1000, eta=0.01, n_epochs=10, lamda=0)
    #classifier.add_layer(n=, input_nodes=)




if __name__ == "__main__":
    main()
