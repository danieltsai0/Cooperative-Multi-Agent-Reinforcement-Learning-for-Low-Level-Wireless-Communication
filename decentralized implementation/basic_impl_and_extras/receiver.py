############################################################
#
# K-NN receiver, outputs reward for transmitter to train on
# Daniel Tsai <daniel_tsai@berkeley.edu>
#
############################################################ 

from util import *

import tensorflow as tf
import numpy as np
import IPython as ipy
import itertools
import matplotlib.pyplot as plt
import time

"""
Source: 
    http://stackoverflow.com/questions/43554819/numpy-or-scipy-find-the-mode-of-a-matrix-of-vectors?noredirect=1#43555673

Inputs:
    a: a numpy array of size (?,num_bits)

Outputs:
    most_frequent_row: the vector that occurred most frequently in the input
"""
def mode_rows(a, label):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row[None]


"""
Inputs:
    data:    numpy array of complex numbers
             shape: (?,)
    labels:  numpy array of bitstrings
             shape: (?, num_bits)
    func:    function to apply to knn labels

Outputs:
    output: numpy array of bitstring guesses 
             for each input data point based 
             on k nearest neighbors
             shape: (?, num_bits)
"""
def knn(k, data, labels, func):
    output = np.empty((0, labels.shape[1]))
    i = 0
    for d in data:
        dist = np.linalg.norm(d-data,axis=1)
        idx = dist.argsort()[1:k+1]
        output = np.r_[output,func(labels[idx,:], labels[i,:])]
        i += 1

    return output

"""
Inputs:
    a: a numpy array of size (?,num_bits)
    label: the correct label of the transmission

Outputs:
    percentage of labels that were correct in a
"""
def percent_correct(a, label):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    a = a.view(void_dt).ravel()
    label = label.view(void_dt).ravel()
    return np.count_nonzero(a==label) / a.shape[0]

"""
This relies on a the transmitter sending a preamble, which was 
agreed upon by both sides beforehand. 
"""
class KnnReceiver():
    def __init__(self, preamble, size_of_episode, n_bits, k, l_or_p):
        self.preamble = preamble
        self.guesses = None
        self.k = k
        self.func = mode_rows if l_or_p else percent_correct

        # values for keeping track of where we are in the preamble
        self.size_of_episode = size_of_episode
        self.index = 0
        self.max_index = (self.preamble.shape[0] // size_of_episode) - 1

    # Receive signal from transmitter
    # Generates guesses, which represents the knn guess for each signal
    # Can change reward to be % of correct labels in k nearest neighbors
    def receive(self, signal):
        data = signal
        print("data.shape:",data.shape)
        labels = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        print("labels.shape:",labels.shape)
        print("receiver index:",self.index)
        self.index += 1 % self.max_index
        self.output = knn(self.k, data, labels, self.func)

    # Transmit guesses back to the transmitter
    def transmit(self):
        print("output.shape:",self.output.shape)
        return self.output

    def visualize(self, iteration):
        raise NotImplementedError