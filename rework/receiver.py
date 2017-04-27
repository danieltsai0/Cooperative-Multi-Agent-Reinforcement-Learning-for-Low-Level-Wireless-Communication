
import tensorflow as tf
import numpy as np
import IPython as ipy
import itertools
import matplotlib.pyplot as plt
import util


class KnnReceiver():
    def __init__(self, preamble, k):
        self.labels = preamble
        self.k = k


    def mode_rows(self, a):
        a = np.ascontiguousarray(a)
        void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
        _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                    return_index=1,return_counts=1)
        largest_count_id = ids[count.argmax()]
        most_frequent_row = a[largest_count_id]
        return most_frequent_row[None]


    def knn(self, data, signal_m=None):

        if signal_m is None:
            signal = data
            func = lambda dist: dist.argsort()[1:self.k+1]
        else:
            signal = signal_m
            func = lambda dist: dist.argsort()[:self.k]

        signal_b = np.empty((0, self.labels.shape[1]))

        for d in signal:
            dist = np.linalg.norm(d - data,axis=1)
            idx = func(dist)
            signal_b = np.r_[signal_b, self.mode_rows(self.labels[idx,:])]

        return signal_b


    def receive(self, *args):
        signal_m, signal_g_m = args[0], None
        if len(args) == 2:
            signal_g_m = args[1]

        return self.knn(signal_m, signal_g_m)


            