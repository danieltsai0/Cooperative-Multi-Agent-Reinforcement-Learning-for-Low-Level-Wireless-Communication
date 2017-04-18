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
#  and pulseshaped samples
#
################################################################################

################################################################################
# CONFIGURATION

MODULATION = '16qam'
ALPHA = 0.5 # roll-off factor
OSF = 16 # oversampling factor = sequence length 
TOTAL_NUMBER_SEQUENCES = 16*15*14 # total number of sequences in datapool
BATCH_SIZE = 128 
TEST_SIZE = 1000 # number of symbols for testing
HIDDEN_DIM = 64 
NUM_LAYERS = 2 # number of layers in the LSTM
OUTPUT_DIM = 2 
INPUT_DIM = 4 # 2: real & imag, 3: add magnitude, 4: add phase

assert INPUT_DIM>=2 #  

ITERATIONS = 5000
LEARNING_RATE = 1e-6
PRINTEVERY = 100 # how often the training error is reported
PLOT = True # plot at every report
DRAW = False # export images
EYE_WIDTH = 4 # width of the eye diagramm in symbols

################################################################################


################################################################################
# COMPUTATIONAL GRAPH

# input dimensions: (#batch, #timestep, data dimension)
sy_input = tf.placeholder(tf.float32, shape=[None, OSF, INPUT_DIM]) 
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
# Phasenoise function

def add_phasenoise(data_shaped, sigma):
    """ adds a sampling offset to the data by shifting the samples within their bin
        for every sequence of samples (oversampled symbol), the functions draws
        from a truncated gaussian (limited to +-25% sample offset) to get a
        sample offset
    """
    num_samples = data_shaped.shape[0]
    data_return = np.copy(data_shaped) 
    clip = OSF/4.0+0.5
    rs=truncnorm.rvs(-clip/sigma, clip/sigma, scale=sigma, loc=0, size=num_samples/OSF)
    rs = np.round(rs).astype(np.int8)
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
data_raw, data_mod, data_os, data_shaped, rrc = generator.get_pulseshaped_data(TOTAL_NUMBER_SEQUENCES, MODULATION, OSF, alpha=ALPHA)
data_shaped = add_phasenoise(data_shaped, sigma=1) # add phase noise to training data
data_featurized = np.stack([data_shaped.real, data_shaped.imag, np.abs(data_shaped), np.angle(data_shaped)], axis = 1)[:,:INPUT_DIM]
data_batches = np.reshape(data_featurized, [-1, OSF, INPUT_DIM])
data_labels = np.stack([data_mod.real, data_mod.imag], axis=1)  

# test data
data_raw, data_mod, data_os, data_shaped, rrc = generator.get_pulseshaped_data(TEST_SIZE, MODULATION, OSF, alpha=ALPHA)
data_plot = np.copy(data_shaped)
data_shaped = add_phasenoise(data_shaped, sigma=1) 
data_featurized = np.stack([data_shaped.real, data_shaped.imag, np.abs(data_shaped), np.angle(data_shaped)], axis = 1)[:,:INPUT_DIM]
test_data= np.reshape(data_featurized, [-1, OSF, INPUT_DIM])
test_labels= np.stack([data_mod.real, data_mod.imag], axis=1)  
test_feed_dict={sy_input: test_data, sy_target: test_labels} 

if (PLOT): # show data for demonstrative purposes
    
    NUM_SHOW = 16 # samples to show 

    # show generated and pulseshaped data 
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,8))
    fig.suptitle("Pulseshaped data", fontsize=20)

    ax1.set_title("Real Part")
    ax1.stem(data_os[0:NUM_SHOW*OSF].real, label="oversampled data", basefmt=" ")
    ax1.plot(data_plot.real, label="pulseshaped data")
    ax1.set(ylabel='magnitude')
    ax1.legend()

    ax2.set_title("Imag Part")
    ax2.stem(data_os[0:NUM_SHOW*OSF].imag, label="oversampled data", basefmt=" ")
    ax2.plot(data_plot.imag, label="pulseshaped data")
    ax2.set(ylabel='magnitude', xlabel='sample index')
    ax2.set_xlim([0, NUM_SHOW*OSF])
    ax2.legend()

    plt.show(block=False)

    # show rrc filter
    fig = plt.figure(figsize=(10,4))
    plt.title('Root-Raised Cosine Filter', fontsize=20)
    ax1 = fig.add_subplot(111)
    ax1.set(ylabel='sample index', xlabel='sample index')
    ax1.set_xlim([0, rrc.shape[0]])
    ax1.plot(rrc)    
    plt.show(block=False)

    # show eye-diagramm 
    num_testsamples = data_plot.shape[0]
    length_samples = OSF*EYE_WIDTH
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,8))
    fig.suptitle("Eye Diagramm", fontsize=20)

    ax1.set_title("Original Samples")
    for i in np.arange(0,num_testsamples, length_samples):
        ax1.plot(data_plot[i:i+length_samples].real, 'b')
    ax1.set(ylabel='magnitude')
    ax1.set_xlim([0, length_samples-1])
    ax1.set_ylim([-1.25, 1.25])
    ax1.set_xticks(np.arange(OSF/2, length_samples, OSF))
    ax1.xaxis.grid(color='red', linestyle='solid')

    ax2.set_title("Noisy Samples")
    for i in np.arange(0,num_testsamples, length_samples):
        ax2.plot(data_shaped[i:i+length_samples].real, 'b')
    ax2.set(ylabel='magnitude', xlabel='sample index')
    ax2.set_xlim([0, length_samples-1])
    ax2.set_ylim([-1.25, 1.25])
    ax2.set_xticks(np.arange(OSF/2, length_samples, OSF))
    ax2.xaxis.grid(color='red', linestyle='solid')

    plt.show(block=False)

    # show filter
    

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

#plt.scatter(data_shaped.real[:64*16], data_shaped.imag[:64*16])
#plt.show()

for i in range(ITERATIONS+1):
    batch = np.random.randint(0, TOTAL_NUMBER_SEQUENCES, BATCH_SIZE)
    input_batch = data_batches[batch,:,:]
    target_batch = data_labels[batch,:]

    feed_dict={sy_input: input_batch, sy_target: target_batch}
    _, train_error = sess.run([minimize, sy_loss], feed_dict)

    if ((i%PRINTEVERY) == 0): 

        values, test_error = sess.run([sy_output, sy_loss], test_feed_dict)
        print("It: %04d | train_loss: %f, test_loss: %f" % (i, train_error, test_error))
       
        if (PLOT):
            plot = ax.scatter(values[:,0], values[:,1], alpha=0.2)
            fig.canvas.draw() 
            if (DRAW): plt.savefig("out/%04d.png" % i)
            if (i != ITERATIONS): plot.remove()

plt.show()


