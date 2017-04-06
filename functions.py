#!/usr/bin/python

##################################################
# This provides a collection of base band functions
# Author: Colin de Vrieze <cdv@berkeley.edu>
##################################################

import numpy as np
from cmath import rect

sq2 = np.sqrt(2)
constellations = {
    'bpsk': np.array([-1+0j, 1+0j]),
    'qpsk': np.array([-1-1j, 1-1j, -1+1j, 1+1j]/sq2),
    '8psk': np.array([-1-1j, -1*sq2, 1j*sq2, -1+1j, -1j*sq2, 1-1j, 1+1j, sq2]/sq2),
    '16qam': np.array([-3+3j, -3+1j, -3-3j, -3-1j,
                        -1+3j, -1+1j, -1-3j, -1-1j,
                        3+3j, 3+1j, 3-3j, 3-1j,
                        1+3j, 1+1j, 1-3j, 1-1j]/(3*sq2))
    }

def modulate():
    """ maps a batch of bits to a given constellation
        INPUT:
            symbols: list or np.array of symbols to be modulated
            constellation: list or np.array of complex-valued constellation points 
        OUTPUT:
            list of complex valued symbols    
    """    
    pass

def modulate_symbol(symbol, constellation):
    assert symbol < constellation.shape
    return constellation[symbol]
modulate = np.vectorize(modulate_symbol, excluded=['constellation'])
modulate.excluded.add(1)

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

