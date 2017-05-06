import argparse
import pickle
import numpy as np 

import util
from receiver import KnnReceiver
from channel import Channel


#### Variables ####
PREAMBLE_LEN = 512 # length of preamble
TEST_LEN = int(1e7) # length of test set
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
# 		   help="path to centroids file")
# args, leftovers = parser.parse_known_args()

# Mapping
# centroids = [util.schemes[n_bits]] + [pickle.load(open(args.cfile[i], "rb")) for i in range(len(args.cfile))]
centroids = [util.schemes[n_bits]] + [pickle.load(open(cfile[i], "rb")) for i in range(len(cfile))]
# Create preamble
preamble = np.random.randint(0,2,[PREAMBLE_LEN,n_bits])
message = np.random.randint(0,2,[TEST_LEN,n_bits])
# Eb/N0
ebn0_values = np.linspace(*EBN0_RANGE)
# Define receiver
receiver = KnnReceiver(preamble, k)
# Other vars
baseline_scheme = util.schemes[n_bits]
output_string = ",".join([str(x) for x in ebn0_values]) + "\n"

# Run computations
for i,centroid in enumerate(centroids):
	print("processing file number:",i)
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
		demod_message = receiver.receive(mod_preamble_n, mod_message_n)
		# Compute BER
		ber = np.sum(np.linalg.norm(demod_message - message, ord=1, axis=1)) / (TEST_LEN*n_bits)
		# Add to list
		ber_vals.append(str(ber))
		print("    EbN0: %8f, ber: %d" % (EbN0, ber))

	output_string += ",".join(ber_vals) + "\n"

with open(FILENAME, "w") as f:
	f.write(output_string)