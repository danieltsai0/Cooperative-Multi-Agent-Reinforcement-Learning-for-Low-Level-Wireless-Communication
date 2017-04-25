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
def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row[None]


"""
Used for demodulating preamble signal.

Inputs:
    data:    numpy array of complex numbers to be demodulated
             shape: (?,)
    labels:  numpy array of bitstrings that correspond to each row in (data)
             shape: (?, num_bits)
    func:    function to apply to knn labels

Outputs:
    output: numpy array of bitstring guesses 
             for each input data point based 
             on k nearest neighbors
             shape: (?, num_bits)
"""
def knn_preamble(k, data, labels, func):
    output = np.empty((0, labels.shape[1]))
    i = 0
    for d in data:
        dist = np.linalg.norm(d-data,axis=1)
        idx = dist.argsort()[1:k+1]
        output = np.r_[output,func(labels[idx,:])]
        i += 1

    return output


"""
Used for interpreting preamble guess signal.

Inputs:
    data:    numpy array of complex numbers to build map
             shape: (?,)
    labels:  numpy array of bitstrings that correspond to each row in (data)
             shape: (?, num_bits)
    signal:  numpy array of complex numbers to be demodulated
             shape: (?,)
    func:    function to apply to knn labels

Outputs:
    output: numpy array of bitstring guesses 
             for each input data point based 
             on k nearest neighbors
             shape: (?, num_bits)
"""
def knn_reward(k, data, labels, signal, func):
    output = np.empty((0, labels.shape[1]))
    i = 0
    for d in signal:
        dist = np.linalg.norm(d-data,axis=1)
        idx = dist.argsort()[:k]
        output = np.r_[output,func(labels[idx,:])]
        i += 1

    return output


"""
This relies on a the transmitter sending a preamble, which was 
agreed upon by both sides beforehand. 
"""
class KnnReceiver():
    def __init__(self, n_bits, k):
        self.preamble_mod = None # Modulated preamble
        self.preamble_bit = None # Bit repr preamble
        self.mod_mod = None # Modulated reward
        self.k = k
        self.func = mode_rows

    """
    Internal function for setting modulated preamble and bit format premable for demodulation use.
    """
    def receive(self, preamble_mod, preamble_bit):
        self.preamble_mod = preamble_mod
        # print("preamble_mod.shape:",self.preamble_mod.shape)
        self.preamble_bit = preamble_bit
        # print("preamble_bit.shape:",preamble_bit.shape)        

    """
    Demodulates modulated preamble signal using Knn
    """
    def generate_guess(self):
        self.output = knn_preamble(self.k, self.preamble_mod, self.preamble_bit, self.func)
        return self.output

    """
    Demodulates modulated preamble guess signal using Knn

    Inputs:
        preamble_g_mod: modulated preamble guess
    """
    def generate_demod(self, preamble_g_mod):
        # print("reward_mod.shape:",reward_mod.shape)
        self.preamble_g_mod = preamble_g_mod
        self.output = knn_reward(self.k, self.preamble_mod, self.preamble_bit, self.preamble_g_mod, self.func)
        return self.output
