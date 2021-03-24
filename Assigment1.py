from load_data import load_batch
import numpy as np

def load_data():
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    validation_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")

    return {'train_data':train_data,
            'validation_data': validation_data,
            'test_data': test_data}

def preprocessing(data):
    mean_x = np.mean(data['train_data']['data'],axis=0)
    std_x = np.std(data['train_data']['data'],axis=0)
    
    return data

def main():
    data = load_data()
    preprocessing(data)


if __name__ == "__main__":
    main()

