import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Data_generator import Data_generator


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
OSF = 8 # oversampling factor = sequence length 
TOTAL_NUMBER_SEQUENCES = 16*15*14 # total number of sequences in datapool
BATCH_SIZE = 32 
TEST_SIZE = 200 # number of symbols for testing
HIDDEN_DIM = 20 
NUM_LAYERS = 2 # number of layers in the LSTM
OUTPUT_DIM = 2 # defines maximum number of output bits
INPUT_DIM = 2 # 2: real & imag, 3: add magnitude, 4: add phase

assert INPUT_DIM>=2 #  

ITERATIONS = 3000
LEARNING_RATE = 1e-3
PRINTEVERY = 100 # how often the training error is reported
PLOT = True # plot at every report

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

# try to add a 1d conv
#conv1 = tf.layers.conv1d(inputs=sy_input,
#                         filters=3,
#                         kernel_size=2)
                         
sy_outputs, sy_state = tf.nn.dynamic_rnn(sy_stacked_lstm, 
                                         sy_input, 
                                         initial_state=sy_initial_state, 
                                         time_major=False)

# we are only interested in the last output
sy_last = tf.unstack(sy_outputs, axis=1)[-1] 

# transform outputs of LSTM to output target (softmax probabilities)
sy_w = tf.Variable(tf.truncated_normal([HIDDEN_DIM, OUTPUT_DIM]), name="output_weight")
sy_b = tf.Variable(tf.constant(0.1, shape=[OUTPUT_DIM]), name="output_bias")
sy_output = tf.matmul(sy_last, sy_w) + sy_b

# calculate cross entropy loss and error indicators
sy_loss = tf.reduce_mean((sy_output-sy_target)**2)
 
minimize = tf.train.AdamOptimizer().minimize(sy_loss)

################################################################################


################################################################################
# EXPERIMENT 

# generate data 
generator = Data_generator()

# training data
data_raw, data_mod, data_os, data_shaped, rrc = generator.get_pulseshaped_data(OSF*TOTAL_NUMBER_SEQUENCES, MODULATION, OSF)
data_featurized = np.stack([data_shaped.real, data_shaped.imag, np.abs(data_shaped), np.angle(data_shaped)], axis = 1)[:,:INPUT_DIM]
data_batches = np.reshape(data_featurized, [-1, OSF, INPUT_DIM])
data_labels = np.stack([data_mod.real, data_mod.imag], axis=1)  

# test data
data_raw, data_mod, data_os, data_shaped, rrc = generator.get_pulseshaped_data(200, MODULATION, OSF)
# add offset
data_shaped = np.roll(data_shaped, 2)

plt.plot(data_shaped.real[:100])
plt.stem(data_os[:100])
plt.show()

data_featurized = np.stack([data_shaped.real, data_shaped.imag, np.abs(data_shaped), np.angle(data_shaped)], axis = 1)[:,:INPUT_DIM]
test_data= np.reshape(data_featurized, [-1, OSF, INPUT_DIM])
test_labels= np.stack([data_mod.real, data_mod.imag], axis=1)  
test_feed_dict={sy_input: test_data, sy_target: test_labels} 

if (PLOT):  
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
        print("It: %d | train_loss: %f, test_loss: %f" % (i, train_error, test_error))
       
        if (PLOT):
            plot = ax.scatter(values[:,0], values[:,1], alpha=1)
            fig.canvas.draw() 
            plt.savefig("out/%04d.png" % i)
            if (i != ITERATIONS): plot.remove()

plt.show()


