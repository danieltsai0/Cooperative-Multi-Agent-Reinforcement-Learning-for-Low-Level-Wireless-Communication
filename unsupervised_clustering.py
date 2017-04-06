#!/usr/bin/python

##################################################
# This script does things
# Author: Colin de Vrieze <cdv@berkeley.edu>
##################################################

##################################################
# Imports

from functions import *
from spectral import *
from k_means import *
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from scipy.sparse.linalg import eigs
from scipy.stats import chi2
from scipy.linalg import eig

##################################################
# settings

SHOW_CONSTELLATION = True     # plot constellation diagram
SHOW_DISTURBANCE = True
SHOW_COVARIANCES = True
SHOW_GRAPH = True

N0 = 0.1 # Noise power density
chisq_likelihood = chi2.isf(1-0.95, 2) # define confidence interval for contour plot
MAX_K = 20 # maximum possible number of constellation points
MODULATION, CONST_POINTS = ('16qam', 16) # define modulation scheme and number of constellation points
NUM_SAMPLES = 1000 # define how many points are sampled

##################################################
# generate data


x = np.random.randint(0,CONST_POINTS,NUM_SAMPLES)
y = modulate(x, constellations[MODULATION])
y = AWGN(np.array(y), N0)
y = delay(y, 3.14/12)

mapper = k_means(MAX_K)
means = mapper.initialize(y, True)

# perform jump method to find clusters
jump, data = mapper.jump_method(y, 100, True, MAX_K)
num_clusters_predicted = np.argmax(jump) # find most likely k value

# extract data from the run with the most likely k value
distortion, means, assign = data[num_clusters_predicted]

# Spectral Analysis
spectral = Spectral_Analyser()
# G = spectral.epsilon_neighborhood_graph(y, 0.1)
G = spectral.knn_graph(y, 10, mutual=True)
L = spectral.unnormalized_laplacian(G)
w, v = eigs(L, k=16)
plt.plot(abs(w)[::-1])

# fig = plt.figure()
# kernelizer = kernel()
# K = kernelizer.kernelize(y, 1)
# print K.shape
# w, v = eigs(K, k=16)
# print (w.shape)
# plt.plot(abs(w))

##################################################
# Plotting


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


##################################
# junk

# draw confidence intervals 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# mean = np.array([2, 2])
# cov = np.array([[5, 2],
                # [3, 5]]) 
# toy = np.random.multivariate_normal(mean, cov, 5000)
# ax.scatter(toy[:,0],toy[:,1])
# 
# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
