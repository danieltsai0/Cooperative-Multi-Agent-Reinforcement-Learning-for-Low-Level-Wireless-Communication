import argparse
import pickle
import numpy as np 
import multiprocessing
import matplotlib.pyplot as plt

from util import *
from receiver import KnnReceiver
from channel import Channel
from ctypes import *


# Run computations
def single_compute(centroid):
    map_func = lambda x: centroid[tuple(x)]
    mean_Es = np.average([np.sum(np.square(x)) for x in list(centroid.values())])
    ber_vals = []
    # EbN0 values
    for EbN0 in ebn0_values:
        # calculate N0 value to meet Eb/N0 requirement and add noise to sample
        EbN0_lin = 10**(0.1*EbN0)
        power = mean_Es/(EbN0_lin*n_bits)
        # Define channel
        channel = Channel(power)
        # Modulate signal
        mod_preamble = np.array(list(map(map_func, preamble)))
        mod_message = np.array(list(map(map_func, message)))
        # Noisy signal
        mod_preamble_n = channel.AWGN(mod_preamble)
        mod_message_n = channel.AWGN(mod_message)
        # Demodulate signal
        demod_message = centroid_mapping(mod_preamble_n, preamble, mod_message_n)
        # Compute BER
        ber = np.sum(np.linalg.norm(demod_message - message, ord=1, axis=1)) / (TEST_LEN*n_bits)
        # Add to list
        ber_vals.append(str(ber))
        # print("    EbN0: %8f, ber: %8f" % (EbN0, ber))

    print("    done")

    with output_string.get_lock():
        output_string.value += (",".join(ber_vals) + "\n").encode('utf-8')
    

if __name__ == "__main__":

    #### Variables ####
    PREAMBLE_LEN = 512 # length of preamble
    TEST_LEN = 100 # length of test set
    EBN0_RANGE = (0, 16, 40) # min[dB], max[dB], #steps
    FILENAME = "output/BER_eval.csv"
    np.random.seed(0)
    n_bits = 4
    k = 3

    cfile = ['output/centroid.pickle','output/centroid1.pickle','output/centroid2.pickle','output/centroid3.pickle',
             'output/centroid4.pickle','output/centroid5.pickle','output/centroid6.pickle','output/centroid7.pickle',
             'output/centroid8.pickle','output/centroid9.pickle']

    # #### Load centroids and generate preamble
    # parser = argparse.ArgumentParser()

    # parser.add_argument("cfile", type=str, nargs='*',
    #          help="path to centroids file")
    # args, leftovers = parser.parse_known_args()

    # Create preamble
    preamble = np.random.randint(0,2,[PREAMBLE_LEN,n_bits])
    message = np.random.randint(0,2,[TEST_LEN,n_bits])
    # Eb/N0
    ebn0_values = np.linspace(*EBN0_RANGE)
    # Other vars
    baseline_scheme = schemes[n_bits]
    # Mapping
    centroids = [baseline_scheme] + [pickle.load(open(cfile[i], "rb")) for i in range(len(cfile))]

    init_string = (",".join([str(x) for x in ebn0_values]) + "\n")
    output_string = multiprocessing.Value("c_char", init_string.encode('utf-8'))
    # output_string.value = bytearray("test",'utf-8')

    print("starting ...")
    
    p = multiprocessing.Pool()
    p.map(single_compute, centroids)

    print(output_string.value)

    # with open(FILENAME, "w") as f:
    #     f.write(output_string)