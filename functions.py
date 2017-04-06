#!/usr/bin/python

##################################################
# This provides a collection of base band functions
# Author: Colin de Vrieze <cdv@berkeley.edu>
##################################################

import numpy as np
from cmath import rect

def AWGN(samples, N0):
    """ AWGN channel
        INPUT:
            samples - np.array of complex samples
            N0 - Noise density
        OUTPUT:
            samples with added gaussian noise
    """
    noise_re = np.random.normal(0, N0/2, samples.shape)
    noise_im = np.random.normal(0, N0/2, samples.shape)
    
    return samples + (noise_re + 1j*noise_im) 

def delay(samples, phi):
    """ constant phase offset
        INPUT:
            samples - np.array of complex samples
            phi - phase offset
        OUTPUT:
            samples with phase offset 
    """
    factor = rect(1, phi)
    return samples * factor 

