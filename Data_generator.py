
############################################################
#
#  Data generator
#  Colin de Vrieze <cdv@berkeley.edu>
#
#  This script creates random binary data and modulates
#+ the data according to a given modulation scheme with
#+ given mapping. It can optionally apply pulse shaping
#+ with given oversampling factor and root-raised cosine
#+ filtering
#  
#  can be run in two ways:
#  1. instanciate a Data_generator object and call
#     get_random_data()
#  2. run from command line and load pickle file
#
############################################################ 

import numpy as np
import matplotlib.pyplot as plt
from cmath import rect
# commpy dependency has to be installed
from commpy.filters import rrcosfilter
sq2 = np.sqrt(2)


class Data_generator(object):
    """ Data generator object which can output modulated data """
   
    # class variable "constellations" defines the constellation points.
    # the points are normalized so that the maximum amplitude is 1
    constellations = {
        'bpsk': (2, np.array([-1+0j, 1+0j])),
        'qpsk': (4, np.array([-1-1j, 1-1j, -1+1j, 1+1j]/sq2)),
        '8psk': (8, np.array([-1-1j, -1*sq2, 1j*sq2, -1+1j, -1j*sq2, 1-1j, 1+1j, sq2]/sq2)),
        '16qam': (16, np.array([-3+3j, -3+1j, -3-3j, -3-1j,
                            -1+3j, -1+1j, -1-3j, -1-1j,
                            3+3j, 3+1j, 3-3j, 3-1j,
                            1+3j, 1+1j, 1-3j, 1-1j]/(3*sq2)))
        }

    def __init__(self, seed=0):
        np.random.seed(seed)
        pass


    def _modulate_symbol(symbol, constellation):
        """ helper function to map one single symbol to its constellation point
            INPUT:
                symbol: integer value of information bits (e.g. 0101 = 5)
                constellation: np.array of constellation points
                    0 gets mapped to first constellation point and so on...
            OUTPUT:
                constellation point for the given input 
        """
        try:
            return constellation[symbol]
        except IndexError as error:
            print("Modulation failed. Most probably the number of information bits\
                   was too high to be modulated with the selected modulation scheme:\n\
                   BSPK: 1 bit, QPSK: 2 bit, 8PSK: 3 bit, 16QAM: 16 bit")
            raise
            return -1


    def get_random_data(self, num, mod):
        """ generates random modulated data
            INPUT:
                num: number of symbols
                mod: modulation scheme: 'bpsk', 'qpsk', ...
            OUTPUT: (data_raw, data_mod)
                data_raw: integer valued np.array with raw (binary) data
                data_mod: complex valued np.array with modulated data
        """

        # check arguments   
        if (num < 1):
            raise ValueError("number of datapoints has to be >0: %d" % num)                

        if (mod not in self.constellations.keys()):
            raise ValueError("Modulation scheme '%s' is not in the list of available modulation schemes: %s" % (mod, str(self.constellations.keys())))
           
        # generate the data
        data_raw = np.random.randint(0,self.constellations[mod][0],num)
        data_mod = self.constellations[mod][1][data_raw]

        return (data_raw, data_mod)

    
    def get_pulseshaped_data(self, num, mod, os, length, alpha=0.35, Ts=1e-6, Fs=1e6):
        """ generates random, oversampled and root-raised cosine filtered data
            INPUT:
                num:
                mod:
                os: oversampling factor
                length: length of the rrc filter
                alpha: roll-off factor
                Ts: symbol period
                Fs: sampling rate
            OUTPUT: (data_raw, data_mod, data_shaped)
                data_raw: integer valued np.array with raw (binary) data
                data_mod: complex valued np.array with modulated data
                data_os: complex valued np.array with oversampled data
                data_shaped: complex valued np.array with pulse-shaped data
                rrc: coefficients of the RRC filter
        """
        Ts = (os/2)*1e-6
        Fs = 1e6
        data_raw, data_mod = self.get_random_data(num, mod)        
        rrc = rrcosfilter(length, alpha, Ts, Fs)[1]
        os_filter = np.zeros(os)
        os_filter[0] = 1        
        data_os = np.kron(data_mod, os_filter)
        # center ramp up and ramp down of filter        
        data_os = np.concatenate([np.zeros(os/2), data_os[:-os/2]])        
        data_shaped = np.convolve(rrc, data_os)
        # center the filter
        data_shaped = data_shaped[length/2:-length/2+1]       
        return (data_raw, data_mod, data_os, data_shaped, rrc)

def _main_showcase():
    
    # for testing the pulseshaped generator
    generator = Data_generator()
    data_raw, data_mod, data_os, data_shaped, rrc = generator.get_pulseshaped_data(1000, 'qpsk', 16, 128)
    print data_raw.shape, data_mod.shape, data_os.shape, data_shaped.shape
    plt.stem(data_os[:12*16].real, axis=0)
    plt.plot(data_shaped[:12*16].real,'r')
    plt.show()
    
    plt.scatter(data_shaped.real[:64], data_shaped.imag[:64], color='red')
    plt.scatter(data_mod.real[:64], data_mod.imag[:64], color='blue')
    plt.show()
    

def main():
    """ runs the data generator from the command line """
    
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    
    # normal operation    
    parser.add_argument("num", type=int, default=1000, 
                               help="number of data points to be generated (1000)")
    parser.add_argument("mod", type=str, default='qpsk', 
                               help="modulation scheme: 'bspk', 'qpsk', '8psk', '16qam' (qpsk))")
    parser.add_argument("--dir", type=str, default='./generated_data', 
                               help="directory/filename to save output to (./generated_data)")
    
    # for pulseshaped data    
    parser.add_argument('--pulse', action='store_true', help="generate pulse shaped data")
    parser.add_argument("-os", type=int, default=16, 
                               help="oversampling factor (power of 2)")
    parser.add_argument("-len", type=int, default=128, 
                               help="filter length (power of 2)")

    args = parser.parse_args()
    num, mod, directory = (args.num, args.mod, args.dir)
    pulse, os, length = (args.pulse, args.os, args.len)
    
    generator = Data_generator()

    # check input arguments
    try:
        pck_file = open(directory+".pck", "w")
    except IOError as error:
        print("Check if the output directory exists")
        raise
    
    # check if os and length are powers of 2
    assert os&(os-1) == 0
    assert length&(length-1) == 0
    
    if (pulse): # generate pulse shaped data
    
        data = generator.get_pulseshaped_data(num, mod, os, length)
        # data = (data_raw, data_mod, data_os, data_shaped, rrc)

        # save the data 
        pickle.dump(data, pck_file)
        pck_file.close()

        print("Generated %d random samples with '%s' modulation." % (num, mod))
        print("Signal is oversampled by a factor of %d and filtered with a rrc filter of length %d" % (os, length))
        print("Data has been saved in %s.pck" % directory)

    else: # generate normal data
        
        data = generator.get_random_data(num, mod)
        
        # save the data 
        pickle.dump(data, pck_file)
        pck_file.close()

        print("Generated %d random samples with '%s' modulation." % (num, mod))
        print("Data has been saved in %s.pck" % directory)
    
# in case this script gets run from command line, run the data generator and safe
#+its output in a pickle file. run python Data_generator.py -h for help
if (__name__ == "__main__"):
    print("Running data generator in command line mode...")
    main() 
    


