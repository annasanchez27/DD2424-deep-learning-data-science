from RNN import RNN
from preprocessing import *
import numpy as np

file = load_data("/Users/annasanchezespunyes/Documents/GitHub/DD2424/lab4/data/goblet_book.txt")
print(file.keys())
model = RNN(file['K'])
h0 = np.zeros((model.m, 1))
x0 = [file['char_to_ind'][char] for char in file['data']]
print(x0[0])
sequence = model.synthesize_seq_chars(h0,x0,200)