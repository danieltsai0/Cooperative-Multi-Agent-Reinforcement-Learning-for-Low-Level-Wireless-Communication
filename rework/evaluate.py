import argparse
import pickle
import numpy as np 
import multiprocessing
import matplotlib.pyplot as plt

import util
from receiver import KnnReceiver
from channel import Channel

def single_compute(centroid, fn):
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
        mod_preamble = np.array(list(map(map_func, PREAMBLE)))
        mod_message = np.array(list(map(map_func, MESSAGE)))
        # Noisy signal
        mod_preamble_n = channel.AWGN(mod_preamble)
        mod_message_n = channel.AWGN(mod_message)
        # Demodulate signal
        demod_message = util.centroid_mapping(mod_preamble_n, PREAMBLE, mod_message_n)
        # Compute BER
        ber = np.sum(np.linalg.norm(demod_message - MESSAGE, ord=1, axis=1)) / (TEST_LEN*n_bits)
        # Add to list
        ber_vals.append(str(ber))
        print("    EbN0: %8f, ber: %8f" % (EbN0, ber))

    print("    done")

    with open(fn, "w") as f:
        f.write(init_string + ",".join(ber_vals) + "\n")


def wrapper_func(param):
    single_compute(**param)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str,
             help="dirname containing centroids")
    args, leftovers = parser.parse_known_args()

    # Process args
    dirname = args.dirname + "/"
    temp = args.dirname.split("_")
    PREAMBLE_LEN = 10000
    TEST_LEN = 1e7
    FILENAME = "BER_eval.csv"
    # Misc settings
    np.random.seed(0)
    n_bits = 4

    # Generate preamble and EbN0 range
    PREAMBLE = np.random.randint(0,2,[PREAMBLE_LEN,n_bits])
    MESSAGE = np.random.randint(0,2,[int(TEST_LEN),n_bits])
    EBN0_RANGE = (0, 16, 40) # min[dB], max[dB], #steps
    

    centroid_dir = [dirname + FILENAME]
    centroid_file = []
    for i in range(5):
        centroid_dir.append(dirname + str(i) + "/" + "agent_1/" + FILENAME)
        centroid_dir.append(dirname + str(i) + "/" + "agent_2/" + FILENAME)
        centroid_file.append(dirname + str(i) + "/" + "agent_1/centroid.pickle")
        centroid_file.append(dirname + str(i) + "/" + "agent_2/centroid.pickle")

    # Eb/N0
    ebn0_values = np.linspace(*EBN0_RANGE)
    # Other vars

    # Mapping
    centroids = [util.schemes[n_bits]] + [pickle.load(open(centroid_file[i], "rb")) for i in range(len(centroid_file))]

    init_string = (",".join([str(x) for x in ebn0_values]) + "\n")

    print("starting ...")

    # Build args
    params = []
    for i in range(len(centroids)):
        run = dict(centroid=centroids[i], 
                   fn=centroid_dir[i])
        params.append(run)

    
    p = multiprocessing.Pool()
    p.map(wrapper_func, params)
