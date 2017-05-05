#!/usr/bin/python

##################################################
# This script tests modulation schemes 
# Author: Colin de Vrieze <cdv@berkeley.edu>
##################################################

##################################################
# Imports

from Data_generator import Data_generator
from functions import *
import numpy as np
import warnings
import argparse

import matplotlib.pyplot as plt

##################################################
# (default) settings

SHOW = True     # enable/disable plots 
NUM_SAMPLES = 1e7 # define how many points are sampled
TRAIN_LENGTH = 512 # define how many points are used for training (preamble)
RESOLUTION = 40 # number of x values
EBN0_RANGE = (0, 16, 40) # min[dB], max[dB], #steps
SEED = 0

##################################################
# parse arguments (arguments override settings)

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, nargs='*', help="path to csv file()")
parser.add_argument("-bl", type=str, default=None, 
                           help="baseline modulation scheme: 'bspk', 'qpsk', '8psk', '16qam'")
args = parser.parse_args()

MODULATION = args.bl
if (MODULATION is None): # No baseline set. evaluate centroids from file
    BASELINE = False
    BITS_PER_SYMBOL = #TODO read from file

else: # evaluate baseline
    BASELINE = True
    MODULATION = '16qam'
    CONST_POINTS = Data_generator().constellations[MODULATION][0]
    BITS_PER_SYMBOL = np.log2(CONST_POINTS)


def evaluate_baseline(test_data):
    """ does a BER measurement with standard modulation schemes """

    # create mapping xor diff -> biterrors 
    error_values = np.array([bin(x).count('1') for x in range(CONST_POINTS)]) 

    x_raw, x = test_data
    means = Data_generator().constellations[MODULATION][1]
    x_recon = np.argmin(np.abs(x[:, None] - means[None, :]), axis=1)

    diff = x_recon^x_raw # bitwise comparison
    bit_errors = np.sum(error_values[diff])
    ber = bit_errors/(NUM_SAMPLES*BITS_PER_SYMBOL)
    return ber


def evaluate_scheme(train_data, test_data):
    """ evaluates a modulation scheme 
        INPUT:
            train_data: (bit data, modulated data)
            test_data: (bit data, modulated data)
        OUTPUT:
            BER
    """

    # x_raw: int data (0, 1,... , 16)
    # x: modulated data : (0.707+0.707j, ...)
    x_raw, x = train_data    
    # train the receiver
    # TODO

    x_raw, x = test_data
    # put test data through receiver
    # TODO
    # provide bitdata in x_recon (0, 3, ...)
    x_recon = 

    # count bit errors- this code is a bit messy 
    diff = x_recon^x_raw # bitwise comparison
    bit_errors = np.sum(error_values[diff])
    ber = bit_errors/(NUM_SAMPLES*BITS_PER_SYMBOL)
    return ber

    
##################################################
# Run analysis 

# generate data for all experiments
np.seed(SEED)
x_train_raw, x_train = #TODO
x_test_raw, x_test = #TODO 
ebn0_values = np.linspace(EBN0_RANGE*)

# generate sequence of Eb/N0 values
bers = []
print("Train points: %d | Test points: %d" % (TRAIN_LENGTH, NUM_SAMPLES))
print("Eb/N0 [dB], BER")

for EbN0 in ebn0_values: # do experiment for all Eb/N0 values

    # calculate N0 value to meet Eb/N0 requirement and add noise to sample
    mean_Es = #TODO: mean of squared amplitudes 
    EbN0_lin = 10**(0.1*EbN0)
    N0 = mean_Es/(EbN0_lin*BITS_PER_SYMBOL)

    y_train = AWGN(x_train, N0)
    y_test = AWGN(x_test, N0)
    train_data = (x_train_raw, y_train)
    test_data = (x_test_raw, y_test) 

    if (BASELINE): # evaluate baseline (standard demodulator)
        ber = evaluate_baseline(test_data)
    else: # evaluate custom scheme 
        ber = evaluate_scheme(train_data, test_data)

    # print record and save ber
    print("%f, %.10f, %d" % (EbN0, ber, num_clusters_predicted))
    bers.append(ber)    

bers = np.array(bers)

##################################################
# Plotting
if (SHOW): # if plotting is enabled:
    fig = plt.figure()
    plt.title('Bit-Error Rate (BER) of demodulation', fontsize=20)
    ax = fig.add_subplot(111)
    ax.plot(ebn0_values, bers, "-bo")
    ax.set(ylabel='BER', xlabel='$E_b/N_0$ (dB)')
    ax.set_yscale('log')
    #ax.set_ylim([0, 1e-8])
    plt.show(block=False)
