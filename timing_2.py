import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Data_generator import Data_generator
from scipy.stats import truncnorm

################################################################################
# 
#  Symbol timimg recovery LSTM 
#  Colin de Vrieze <cdv@berkeley.edu>
#
#  This LSTM tries to recover symbols from a sequence of oversampled
#  and pulseshaped samples. In contrast to version 1 it considers 
#  and episode length of 2 Ts to incorporate signal transitions
#
################################################################################

################################################################################
# CONFIGURATION

MODULATION = '16qam'
ALPHA = 0.5 # roll-off factor
OSF = 8 # oversampling factor 
SEQ_LENGTH = 2*OSF # sequence length
TOTAL_NUMBER_SEQUENCES = 16*15*14 # total number of sequences in datapool
BATCH_SIZE = 256 
TEST_SIZE = 1000 # number of symbols for testing
HIDDEN_DIM = 40 
NUM_LAYERS = 2 # number of layers in the LSTM
OUTPUT_DIM = 2 
INPUT_DIM = 4 # 2: real & imag, 3: add magnitude, 4: add phase

assert INPUT_DIM>=2 #  

ITERATIONS = 3000
LEARNING_RATE = 1e-6
PRINTEVERY = 50 # how often the training error is reported
PLOT = True # plot at every report
DRAW = False # export images
NUM_EYE = 100 # number of sequences for eye diagramm

################################################################################


################################################################################
# COMPUTATIONAL GRAPH

# input dimensions: (#batch, #timestep, data dimension)
sy_input = tf.placeholder(tf.float32, shape=[None, SEQ_LENGTH, INPUT_DIM]) 
sy_target = tf.placeholder(tf.float32, shape= [None, 2])
sy_batchsize = tf.shape(sy_input)[0]

sy_lstm = tf.contrib.rnn.LSTMCell(HIDDEN_DIM)
sy_stacked_lstm = tf.contrib.rnn.MultiRNNCell([sy_lstm]*NUM_LAYERS)
sy_initial_state = sy_state = sy_stacked_lstm.zero_state(sy_batchsize, dtype=tf.float32)

# Version 1: don't unroll LSTM
sy_outputs, sy_state = tf.nn.dynamic_rnn(sy_stacked_lstm, 
                                         sy_input, 
                                         initial_state=sy_initial_state, 
                                         time_major=False)

# we are only interested in the last output
sy_last = tf.unstack(sy_outputs, axis=1)[-1] 

# transform outputs of LSTM to output target (softmax probabilities)
sy_w = tf.Variable(tf.truncated_normal([HIDDEN_DIM, OUTPUT_DIM]), name="output_weight")
sy_b = tf.Variable(tf.constant(0.0, shape=[OUTPUT_DIM]), name="output_bias")
sy_output = tf.matmul(sy_last, sy_w) + sy_b

# calculate cross entropy loss and error indicators
sy_loss = tf.reduce_mean((sy_output-sy_target)**2)
 
minimize = tf.train.AdamOptimizer().minimize(sy_loss)

################################################################################

################################################################################
# data functions

def stream_to_batch(data_shaped, sigma=None):
    """ takes in a stream of data and arranges it into batches of SEQ_LENGTH.
        If sigma is set, it adds a sampling offset to the data by shifting the samples within their bin.
        for every sequence the functions draws
        from a truncated gaussian (limited to +-25% sample offset) to get a
        sample offset
    """
    num_samples = data_shaped.shape[0]

    # add padding to data sequence
    data_shaped = np.concatenate([np.zeros(OSF/2), data_shaped, np.zeros(OSF/2)], axis=0)
    data_return = np.zeros([num_samples/OSF, SEQ_LENGTH], dtype=np.complex64)

    # parameters for noise | rs contains normal distributed sampling offsets
    clip = OSF/4.0+0.5
    rs = np.zeros(num_samples/OSF, dtype=np.int8)
    if (sigma is not None):
        rs = truncnorm.rvs(-clip/sigma, clip/sigma, scale=sigma, loc=0, size=num_samples/OSF)
        rs = np.round(rs).astype(np.int8) 
        rs[0] = 0 

    for i,k in enumerate(np.arange(0, num_samples-OSF, OSF)):
        data_return[i,:] = data_shaped[k+rs[i]:k+rs[i]+SEQ_LENGTH]
    
    return data_return 


def add_phasenoise(data_shaped, sigma):
    num_samples = data_shaped.shape[0]
    data_return = np.copy(data_shaped) 
    for step,i in enumerate(np.arange(OSF, num_samples-OSF, OSF)):
        r = rs[step]
        data_return[i:i+OSF+1] = data_shaped[i+r:i+r+OSF+1]       

    return data_return 

################################################################################
# EXPERIMENT 
#
# the script generates two sets of data: one for training, one for testing.
# The script adds sampling offsets (phase noise) with normal distribution
# and featurizes the data (real, imag, mag, phase). Next, it gets shaped into
# batches of sequences with length equal to the oversampling factor (1 sequence
# = all samples that belong to a single symbol)

# generate data 
generator = Data_generator()

# training data
data_raw, data_mod, data_os, data_shaped_stream, rc = generator.get_pulseshaped_data(TOTAL_NUMBER_SEQUENCES, MODULATION, OSF, alpha=ALPHA)
data_shaped_batches = stream_to_batch(data_shaped_stream, sigma=1)
data_featurized = np.stack([data_shaped_batches.real, 
                            data_shaped_batches.imag,
                            np.abs(data_shaped_batches),
                            np.angle(data_shaped_batches)], axis = 2)[:,:,:INPUT_DIM]

data_training = data_featurized
data_labels = np.stack([data_mod.real, data_mod.imag], axis=1)  

# test data
data_raw, data_mod, data_os, data_shaped_stream, rc = generator.get_pulseshaped_data(TEST_SIZE, MODULATION, OSF, alpha=ALPHA)
data_shaped_batches = stream_to_batch(data_shaped_stream, sigma=1)
data_shaped_clean= stream_to_batch(data_shaped_stream)
data_featurized = np.stack([data_shaped_batches.real, 
                            data_shaped_batches.imag,
                            np.abs(data_shaped_batches),
                            np.angle(data_shaped_batches)], axis = 2)[:,:,:INPUT_DIM]

test_data = data_featurized
test_labels= np.stack([data_mod.real, data_mod.imag], axis=1)  
test_data[0,:,:], test_labels[0,:] = test_data[1,:,:], test_labels[1,:]
test_feed_dict={sy_input: test_data, sy_target: test_labels} 

if (PLOT): # show data for demonstrative purposes
    
    NUM_SHOW = 16 # samples to show 

    # show generated and pulseshaped data 
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,8))
    fig.suptitle("Pulseshaped data", fontsize=20)

    ax1.set_title("Real Part")
    ax1.stem(data_os[0:NUM_SHOW*OSF].real, label="oversampled data", basefmt=" ")
    ax1.plot(data_shaped_stream[:NUM_SHOW*OSF].real, label="pulseshaped data")
    ax1.set(ylabel='magnitude')
    ax1.legend()

    ax2.set_title("Imag Part")
    ax2.stem(data_os[0:NUM_SHOW*OSF].imag, label="oversampled data", basefmt=" ")
    ax2.plot(data_shaped_stream[:NUM_SHOW*OSF].imag, label="pulseshaped data")
    ax2.set(ylabel='magnitude', xlabel='sample index')
    ax2.set_xlim([0, NUM_SHOW*OSF])
    ax2.legend()

    plt.show(block=False)

    # show rc filter
    fig = plt.figure(figsize=(10,4))
    plt.title('Root-Raised Cosine Filter', fontsize=20)
    ax1 = fig.add_subplot(111)
    ax1.set(ylabel='sample index', xlabel='sample index')
    ax1.set_xlim([0, rc.shape[0]])
    ax1.plot(rc)    
    plt.show(block=False)

    # show eye-diagramm 
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,8))
    fig.suptitle("Eye Diagramm", fontsize=20)

    ax1.set_title("Original Samples")
    for i in np.arange(1,NUM_EYE):
        ax1.plot(data_shaped_clean[i,:].real, 'b')
    ax1.set(ylabel='magnitude')
    ax1.set_xlim([0, SEQ_LENGTH-1])
    ax1.set_ylim([-1.25, 1.25])

    ax2.set_title("Noisy Samples")
    for i in np.arange(1,NUM_EYE):
        ax2.plot(data_shaped_batches[i,:].imag, 'b')
    ax1.set(ylabel='magnitude')
    ax1.set_xlim([0, SEQ_LENGTH-1])
    ax1.set_ylim([-1.25, 1.25])

    plt.show(block=False)
    

if (PLOT): # show convergence
    fig = plt.figure()
    plt.title('Constellation Diagram', fontsize=20)
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set(ylabel='imaginary part', xlabel='real part')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    plt.show(block=False)


# train!
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# calculate baseline
err_bl = np.mean((test_data[:,8,:2]-test_labels)**2)

for i in range(ITERATIONS+1):
    batch = np.random.randint(0, TOTAL_NUMBER_SEQUENCES, BATCH_SIZE)
    input_batch = data_training[batch,:,:]
    target_batch = data_labels[batch,:]

    feed_dict={sy_input: input_batch, sy_target: target_batch}
    _, train_error = sess.run([minimize, sy_loss], feed_dict)

    if ((i%PRINTEVERY) == 0): 

        values, test_error = sess.run([sy_output, sy_loss], test_feed_dict)
        print("It: %04d | train_loss: %f, test_loss: %f, baseline: %f" % (i, train_error, test_error, err_bl))
       
        if (PLOT):
            plot = ax.scatter(values[1:,0], values[1:,1], alpha=0.2)
            fig.canvas.draw() 
            if (DRAW): plt.savefig("out/%04d.png" % i)
            if (i != ITERATIONS): plot.remove()

plt.show()


