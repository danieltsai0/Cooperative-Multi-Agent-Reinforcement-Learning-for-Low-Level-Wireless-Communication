#!/usr/bin/python

##################################################
# This script does things
# Author: Colin de Vrieze <cdv@berkeley.edu>
##################################################

##################################################
# Imports

from Data_generator import Data_generator
from functions import *
from spectral import *
from k_means import *
import numpy as np
import warnings
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from scipy.sparse.linalg import eigs
from scipy.stats import chi2, mode
from scipy.linalg import eig

##################################################
# (default) settings

SHOW_CONSTELLATION = True     # plot constellation diagram
SHOW_DISTURBANCE = True       # plot disturbance analysis
SHOW_COVARIANCES = True       # plot confidence intervals
SHOW_GRAPH = False            # plot adjacency graph

N0 = 0.2 # Noise power density
chisq_likelihood = chi2.isf(1-0.95, 2) # define confidence interval for contour plot
MAX_K = 20 # maximum possible number of constellation points
MODULATION, CONST_POINTS = ('qpsk', 4) # define modulation scheme and number of constellation points
NUM_SAMPLES = 1e6 # define how many points are sampled
TRAIN_LENGTH = 1000 # define how many samples are used for training
NUM_ITERATIONS = 50 # number of iterations to run x-means algorithm
RESOLUTION = 40 # number of x values
EBN0 = 0 # Eb/N0 value for single run

##################################################
# parse arguments (arguments override settings)

parser = argparse.ArgumentParser()
parser.add_argument("mod", type=str, default=MODULATION, 
                           help="modulation scheme: 'bspk', 'qpsk', '8psk', '16qam'")
parser.add_argument("samples", type=int, default=NUM_SAMPLES, 
                           help="number of samples for testing")
parser.add_argument("train", type=int, default=TRAIN_LENGTH, 
                           help="number of training samples")
parser.add_argument("-res", type=int, default=RESOLUTION,
                           help="number of x values")
parser.add_argument("-ebn0", type=int, default=EBN0,
                           help="Eb/N0 value for single run [dB]")
parser.add_argument("-iter", type=int, default=NUM_ITERATIONS,
                           help="number of iterations for k-means")
parser.add_argument("--dir", type=str, default='./out', 
                           help="directory/filename to save output to (./out)")
parser.add_argument('--plot', action='store_true', help="show plots?")
parser.add_argument('--baseline', action='store_true', help="evaluate baseline?")
parser.add_argument('--sweep', action='store_true', help="sweep Eb/N0 values")
args = parser.parse_args()

SHOW = args.plot
SWEEP = args.sweep
MODULATION = args.mod
CONST_POINTS = Data_generator().constellations[MODULATION][0]
BITS_PER_SYMBOL = np.log2(CONST_POINTS)
TRAIN_LENGTH = args.train
NUM_SAMPLES = args.samples
RESOLUTION = args.res
BASELINE = args.baseline
EBN0 = 10**(args.ebn0/20)
NUM_ITERATIONS = args.iter

def evaluate_baseline(test_data):
    """ does a BER measurement with standard modulation schemes """

    error_values = np.array([bin(x).count('1') for x in range(MAX_K)]) # create mapping xor diff -> biterrors 

    x_raw, x = test_data
    means = Data_generator().constellations[MODULATION][1]
    x_recon = np.argmin(np.abs(x[:, None] - means[None, :]), axis=1)

    diff = x_recon^x_raw # bitwise comparison
    bit_errors = np.sum(error_values[diff])
    ber = bit_errors/(NUM_SAMPLES*BITS_PER_SYMBOL)
    return ber


def do_experiment(train_data, test_data):
    """ conducts single experiment 
        INPUT:
            train_data: (data_raw, data)
            test_data: (data_raw, data)
            bool learn: use unsupervised learning or classic receiver?
        OUTPUT:
            BER
    """

    x_raw, x = train_data    

    # STEP 1: find clusters
    # perform jump method to find clusters
    mapper = k_means(MAX_K)
    jump, data = mapper.jump_method(x, NUM_ITERATIONS, True, MAX_K)
    num_clusters_predicted = np.argmax(jump) # find most likely k value

    # extract data from the run with the most likely k value
    distortion, means, assign_train = data[num_clusters_predicted]


    # STEP 2: find mapping    
    # modes saves the bit-values for each cluster mean as integer. It is found
    #+ by taking the mode of the symbols which have been assigned to each cluster
    modes = np.zeros(means.size, dtype = np.int8) 
    error_values = np.array([bin(x).count('1') for x in range(MAX_K)]) # create mapping xor diff -> biterrors 

    for mean in range(num_clusters_predicted):
        modes[mean] = mode(x_raw[assign_train == mean])[0][0]
    
    # STEP 3: evaluate BER
    x_raw, x = test_data
    assign = np.argmin(np.abs(x[:, None] - means[None, :]), axis=1)
    x_recon = modes[assign]

    # count bit errors- this code is a bit messy 
    diff = x_recon^x_raw # bitwise comparison
    bit_errors = np.sum(error_values[diff])
    ber = bit_errors/(NUM_SAMPLES*BITS_PER_SYMBOL)
    return distortion, means, assign_train, jump, num_clusters_predicted, ber

def do_random_experiment(EbN0):
     
    generator = Data_generator(0)
    x_train_raw, x_train = generator.get_random_data(TRAIN_LENGTH, MODULATION)
    x_test_raw, x_test = generator.get_random_data(NUM_SAMPLES, MODULATION)

    mean_Es = np.mean(np.abs(Data_generator().constellations[MODULATION][1]))
    N0 = mean_Es/(EbN0*BITS_PER_SYMBOL)
    y_train = AWGN(x_train, N0)
    y_test = AWGN(x_test, N0)
    train_data = (x_train_raw, y_train)
    test_data = (x_test_raw, y_test) 

    distortion, means, assign_train, jump, num_clusters_predicted, ber = do_experiment(train_data, test_data)
    y = y_train

    return y, distortion, means, assign_train, jump, num_clusters_predicted, ber

    
##################################################
# Run analysis 
if (SWEEP): # sweep Eb/N0 values

    # generate data for all experiments
    generator = Data_generator(0)
    x_train_raw, x_train = generator.get_random_data(TRAIN_LENGTH, MODULATION)
    x_test_raw, x_test = generator.get_random_data(NUM_SAMPLES, MODULATION)
    ebn0_values = np.logspace(-0.5, 0.5, RESOLUTION)

    # generate sequence of Eb/N0 values
    ebn0_values_dB = 20*np.log10(ebn0_values)
    bers = []
    print(MODULATION, " Train points: %d | Test points: %d" % (TRAIN_LENGTH, NUM_SAMPLES))
    print("Eb/N0 [dB], BER, num_clusters")

    for EbN0 in ebn0_values: # do experiment for all Eb/N0 values

        # calculate N0 value to meet Eb/N0 requirement and add noise to sample
        mean_Es = np.mean(np.abs(Data_generator().constellations[MODULATION][1]))
        N0 = mean_Es/(EbN0*BITS_PER_SYMBOL)
        y_train = AWGN(x_train, N0)
        y_test = AWGN(x_test, N0)
        train_data = (x_train_raw, y_train)
        test_data = (x_test_raw, y_test) 

        if (BASELINE): # evaluate baseline (standard demodulator)
            num_clusters_predicted = 0
            ber = evaluate_baseline(test_data)
        else: # conduct experiment 
            distortion, means, assign, jump, num_clusters_predicted, ber = do_experiment(train_data, test_data)

        # print record and save ber
        print("%f, %.10f, %d" % (20*np.log10(EbN0), ber, num_clusters_predicted))
        bers.append(ber)    
    
    bers = np.array(bers)
    bers_log = np.log10(bers)

    # plot
    fig = plt.figure()
    plt.title('Bit-Error Rate (BER) of demodulation', fontsize=20)
    ax = fig.add_subplot(111)
    ax.plot(ebn0_values_dB, bers, "-bo")
    ax.set(ylabel='BER', xlabel='$E_b/N_0$ (dB)')
    ax.set_yscale('log')
    ax.set_xlim([-10, 10])
    plt.show()

else: # run single experiment

    y, distortion, means, assign, jump, num_clusters_predicted, ber = do_random_experiment(EBN0)
    print("BER: ", ber)

##################################################
# Plotting

if (SHOW): # if plotting is enabled:

    if (SHOW_COVARIANCES): # calculate covariance matricies
        covs = []
        for k in range(num_clusters_predicted): 
            y_normalized = np.array([(y[assign==k]-means[k]).real, (y[assign==k]-means[k]).imag]) 
            covs.append(np.cov(y_normalized, ddof=2))

    if (SHOW_CONSTELLATION): 

        # plot data in constellation diagram
        fig = plt.figure()
        plt.title('Constellation Diagram (original)', fontsize=20)
        ax = fig.add_subplot(111)
        ax.scatter(y.real, y.imag, color='blue', alpha=0.3)
        ax.axis('equal')
        ax.set(ylabel='imaginary part', xlabel='real part')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

        if (SHOW_GRAPH):
            if G is None:
                warnings.warn("No similarity Graph defined")
            else:
                for x0 in range(0, G.shape[0]):
                    for x1 in range(x0+1, G.shape[1]):
                        if (G[x0,x1] != 0):
                            line = Line2D([y[x0].real, y[x1].real], [y[x0].imag, y[x1].imag], color='green')
                            ax.add_artist(line)                        

        # plot clustered constellation
        fig = plt.figure()
        plt.title('Constellation Diagram (clustered)', fontsize=20)
        ax = fig.add_subplot(111)
        colors = cm.rainbow(np.linspace(0, 1, num_clusters_predicted))
        for k in range(num_clusters_predicted):
            ax.scatter(y[assign==k].real, y[assign==k].imag, color=colors[k], alpha=0.3)
            ax.scatter(means[k].real, means[k].imag, color=colors[k], marker='o', s=100)
            
            if (SHOW_COVARIANCES): # show confidence intervals 
                w, v = eig(covs[k])
                width = 2*np.sqrt(w[0].real*chisq_likelihood)
                height = 2*np.sqrt(w[1].real*chisq_likelihood)
                angle = np.rad2deg(np.arctan(v[1,0]/v[0,0]))
                ellip = Ellipse(xy=[means[k].real, means[k].imag], width=width, height=height, angle=angle, edgecolor=colors[k], fc='None', lw=1, alpha=0.5) 
                ax.add_artist(ellip) 

        ax.axis('equal')
        ax.set(ylabel='imaginary part', xlabel='real part')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
       

    if (SHOW_DISTURBANCE):
        fig = plt.figure()
        plt.title('Disturbance Analysis', fontsize=20)
        ax = fig.add_subplot(111)
        ax.plot(jump)
        ax.set(xlabel='number of clusters', ylabel='$d[i]-d[i-1]$')

    plt.show()


################################################################################
### CLIPBOARD


##################################################
# Spectral Analysis

# EXPERIMENTAL: do spectral analysis
if (SHOW_GRAPH):
    # Spectral Analysis
    spectral = Spectral_Analyser()
    # G = spectral.epsilon_neighborhood_graph(y, 0.1)
    G = spectral.knn_graph(y, 10, mutual=False)
    L = spectral.unnormalized_laplacian(G)
    w, v = eigs(L, k=16)
    plt.plot(abs(w)[::-1])
