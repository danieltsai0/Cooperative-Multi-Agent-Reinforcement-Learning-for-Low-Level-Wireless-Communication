import numpy as np
from cmath import rect
sq2 = np.sqrt(2)

############################################################
#
#  Data generator
#  Colin de Vrieze <cdv@berkeley.edu>
#
#  This script creates random binary data and modulates
#+ the data according to a given modulation scheme with
#+ given mapping.
#  
#  can be run in two ways:
#  1. instanciate a Data_generator object and call
#     get_random_data()
#  2. run from command line and load pickle file
#
############################################################ 

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

    def __init__(self):
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

    def modulate(self):
        """ maps a batch of bits to a given constellation (function stub)
            INPUT:
                symbols: np.array of symbols to be modulated
                constellation: list or np.array of complex-valued constellation points 
            OUTPUT:
                list of complex valued symbols    
        """
        pass
    # fills in the functionality for the function stub defined above
    modulate = np.vectorize(_modulate_symbol, excluded=['constellation'])
    modulate.excluded.add(1)


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
        data_mod = self.modulate(data_raw, self.constellations[mod][1])

        return (data_raw, data_mod)


def main():
    """ runs the data generator from the command line """
    
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("num", type=int, default=1000, 
                               help="number of data points to be generated (1000)")
    parser.add_argument("mod", type=str, default='qpsk', 
                               help="modulation scheme: 'bspk', 'qpsk', '8psk', '16qam' (qpsk))")
    parser.add_argument("--dir", type=str, default='./generated_data', 
                               help="directory/filename to save output to (./generated_data)")
    args = parser.parse_args()
    num, mod, directory = (args.num, args.mod, args.dir)
    generator = Data_generator()

    try:
        pck_file = open(directory+".pck", "w")
    except IOError as error:
        print("Check if the output directory exists")
        raise

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
    


