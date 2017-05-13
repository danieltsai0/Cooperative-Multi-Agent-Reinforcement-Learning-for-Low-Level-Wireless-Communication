
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

    """
	Receives one or two signals and demodulates them. If one signal,
	it will demodulate it using KNN of the preamble labels. If two
	signals, it will demodulate the second signal using the KNN of 
	the preamble, using the first signal.
    """
    def receive(self, *args):
        signal_m, signal_g_m = args[0], None
        if len(args) == 2:
            signal_g_m = args[1]

        return util.knn(self.k, signal_m, self.labels, signal_g_m)


            