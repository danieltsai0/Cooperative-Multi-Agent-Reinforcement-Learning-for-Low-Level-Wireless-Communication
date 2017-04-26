import tensorflow as tf
import numpy as np
from Data_generator import Data_generator

################################################################################
# 
#  receiver-LSTM
#  Colin de Vrieze <cdv@berkeley.edu>
#
#  This LSTM tries to learn the number of constellation points from
#+ sequences of complex numbers. It learns from batches of number
#+ sequences and outputs softmax probabilities over the number of 
#+ constellation points (only 2, 4, 8 and 16 are implemented)
#
################################################################################

################################################################################
# CONFIGURATION

MODULATION, BITS_PER_SYMBOL = ('16qam', 4)
SEQUENCE_LENGTH = 100 # number of symbols per sequence
TOTAL_NUMBER_SEQUENCES = 1000 # total number of sequences in datapool
BATCH_SIZE = 50 
HIDDEN_DIM = 20
NUM_LAYERS = 2 # number of layers in the LSTM
OUTPUT_DIM = 16 # defines maximum number of output bits

ITERATIONS = 100000
LEARNING_RATE = 1e-6
PRINTEVERY = 500 # how often the training error is reported
################################################################################


################################################################################
# COMPUTATIONAL GRAPH

# input dimensions: (#batch, #timestep, data dimension)
sy_input = tf.placeholder(tf.float32, shape=[None, SEQUENCE_LENGTH, 2]) 
sy_target = tf.placeholder(tf.int64, shape= [None])
sy_target_one_hot = tf.one_hot(sy_target, depth=OUTPUT_DIM) 

sy_lstm = tf.contrib.rnn.LSTMCell(HIDDEN_DIM)
sy_stacked_lstm = tf.contrib.rnn.MultiRNNCell([sy_lstm]*NUM_LAYERS)
sy_initial_state = sy_state = sy_stacked_lstm.zero_state(BATCH_SIZE, dtype=tf.float32)

# Version 1: don't unroll LSTM
sy_outputs, sy_state = tf.nn.dynamic_rnn(sy_stacked_lstm, 
                                         sy_input, 
                                         initial_state=sy_initial_state, 
                                         time_major=False)

# we are only interested in the last output
sy_last = tf.unstack(sy_outputs, axis=1)[-1] 

# Version 2: unroll LSTM
# with tf.variable_scope("RNN"):
    # for i in range(SEQUENCE_LENGTH):
        # if (i>0): tf.get_variable_scope().reuse_variables()
        # sy_output, sy_state = sy_lstm(sy_input[:,i,:], sy_state)
# sy_last = sy_output

print ("sy_last: ", sy_last)

# transform outputs of LSTM to output target (softmax probabilities)
sy_w = tf.Variable(tf.truncated_normal([HIDDEN_DIM, OUTPUT_DIM]), name="output_weight")
sy_b = tf.Variable(tf.constant(0.1, shape=[OUTPUT_DIM]), name="output_bias")
sy_softmax_probs= tf.nn.softmax(tf.matmul(sy_last, sy_w) + sy_b)

# calculate cross entropy loss and error indicators
sy_cross_entropy = -tf.reduce_sum(
                    sy_target_one_hot*tf.log(tf.clip_by_value(sy_softmax_probs,1e-10,1.0)))

sy_prediction = tf.argmax(sy_softmax_probs, axis=1)
print ("sy_prediction: ", sy_prediction)

sy_error = tf.reduce_mean(tf.cast(tf.not_equal(sy_prediction, sy_target), tf.float32))
minimize = tf.train.AdamOptimizer().minimize(sy_cross_entropy)

################################################################################


################################################################################
# EXPERIMENT 

# generate data 

generator = Data_generator()
constellations = generator.constellations.keys() # available constellations
num_constellations = len(constellations)
sequence_table = None

# check if the total number of sequences is dividable by the number of constellations
if (TOTAL_NUMBER_SEQUENCES%num_constellations == 0):
    seq_per_constel = TOTAL_NUMBER_SEQUENCES/num_constellations
else:
    raise ValueError("Total number of sequences has to be dividable by the number\
                      of different modulation schemes! %d vs. %d" %
                      (TOTAL_NUMBER_SEQUENCES, num_constellations))

# generate equal amounts of data sequences for each modulation scheme
for constellation in constellations:
    data_raw, data_modulated = generator.get_random_data(seq_per_constel*SEQUENCE_LENGTH, constellation)
    data = np.array([data_modulated.real, data_modulated.imag]) 
    data = np.reshape(data, [seq_per_constel, SEQUENCE_LENGTH, 2])
    target = generator.constellations[constellation][0] # number of constellation points 
    target = np.ones(data.shape[0])*target
    
    if (sequence_table is not None): # collect sequences
        sequence_table = np.concatenate((sequence_table, data), axis=0)
        target_table = np.concatenate([target_table, target], axis=0)
    else:
        sequence_table = data
        target_table = target 


# shuffle data
perm = np.arange(TOTAL_NUMBER_SEQUENCES)
np.random.shuffle(perm)
sequence_table = sequence_table[perm,:,:]
target_table = target_table[perm]
   
# train!
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(ITERATIONS):
    batch = np.random.randint(0, TOTAL_NUMBER_SEQUENCES, BATCH_SIZE)
    input_batch = sequence_table[batch,:,:]
    target_batch = target_table[batch]

    feed_dict={sy_input: input_batch, sy_target: target_batch}

    _, error = sess.run([minimize, sy_error], feed_dict)
    if ((i%PRINTEVERY) == 0): 
        print("It: %d | err: %f" % (i, error))

################################################################################


