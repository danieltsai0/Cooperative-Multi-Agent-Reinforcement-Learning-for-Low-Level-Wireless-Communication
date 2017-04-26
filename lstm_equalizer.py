import tensorflow as tf
import numpy as np
import scipy.signal as signal
import pickle
import sys
from numpy.lib.stride_tricks import as_strided
from Data_generator import Data_generator

################################################################################
# 
#  equalizer-LSTM
#  Albert Pham <linh.pham@berkeley.edu>
#  Colin de Vrieze <cdv@berkeley.edu>
#
#  This LSTM tries to learn to undo the effects of channel noise from
#+ sequences of noisy complex numbers. It learns from batches of number
#+ sequences and outputs denoised complex numbers. 
#
################################################################################

################################################################################
# CONFIGURATION
def main(CONSTELLATION, DELAY_LENGTH, DATA_FILE, MODEL_FILE):
    ITERATIONS = 1000
    LEARNING_RATE = 1e-6

    SEQUENCE_LENGTH = 100 # number of symbols per sequence
    TOTAL_NUMBER_SEQUENCES = 1000 # total number of sequences in datapool
    BATCH_SIZE = 50
    HIDDEN_DIM = 20
    NUM_LAYERS = 2 # number of layers in the LSTM
    OUTPUT_DIM = 2 # defines maximum number of output bits

    DELAY = np.random.normal(0, 1/np.sqrt(2), (2, DELAY_LENGTH))
    DELAY = np.array([[complex(x, y) for (x, y) in zip(DELAY[0,:], DELAY[1,:])]])

    try:
        assert(DELAY_LENGTH == DELAY.shape[1])
    except:
        print(DELAY.shape, DELAY.size)
        sys.exit()

#PRINTEVERY = 50 # how often the training error is reported
################################################################################
#
# COMPUTATIONAL GRAPH
###############################################################################
    # input dimensions: (#batch, #timestep, data dimension)
    sy_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DELAY_LENGTH, 2]) 
    sy_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, 2])

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

    # transform outputs of LSTM to output target (equalized sequence)
    sy_w_real = tf.Variable(tf.truncated_normal([HIDDEN_DIM, 1]), name="output_weight_real")
    sy_b_real = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias_real")
    sy_output_real = tf.matmul(sy_last, sy_w_real) + sy_b_real
    sy_w_imag = tf.Variable(tf.truncated_normal([HIDDEN_DIM, 1]), name="output_weight_imag")
    sy_b_imag = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias_imag")
    sy_output_imag = tf.matmul(sy_last, sy_w_imag) + sy_b_imag
    sy_output = tf.stack([sy_output_real, sy_output_imag], axis=2)

    try:
        assert(sy_output.shape == sy_target.shape)
    except:
        print("output shape failed", sy_output.shape, sy_target.shape)
        sys.exit()

    sy_error = sy_output - sy_target
    sy_error = tf.reduce_mean(sy_error**2)

    minimize = tf.train.AdamOptimizer().minimize(sy_error)

################################################################################


################################################################################
# EXPERIMENT 
# generate data 
    try:
        npz = np.load(open(DATA_FILE + ".npz", "rb"))
        data, target = npz['data'], npz['target']
        print('Loading data')
        npz.close()

    except:
        print('Generating data')
        generator = Data_generator()

        # generate equal amounts of data sequences for each modulation scheme
        data_raw, data_modulated = generator.get_random_data(SEQUENCE_LENGTH*TOTAL_NUMBER_SEQUENCES, CONSTELLATION)

        target = np.reshape(data_modulated, [TOTAL_NUMBER_SEQUENCES, SEQUENCE_LENGTH])
        target = np.stack([target.real, target.imag], 2) 
        data = np.reshape(data_modulated, [TOTAL_NUMBER_SEQUENCES, SEQUENCE_LENGTH])

        data_delay = signal.convolve2d(DELAY, data)[:, DELAY_LENGTH-1:]

        try: assert(data_delay.shape[1] == SEQUENCE_LENGTH)
        except: print("Test 1 failed", data_delay.shape); sys.exit()
    
        data_delay = np.stack([data_delay.real, data_delay.imag], 2) 

        try: 
            assert(data_delay.shape == (TOTAL_NUMBER_SEQUENCES, SEQUENCE_LENGTH, 2))
        except:
            print("Test 2 failed", data_delay.shape, [TOTAL_NUMBER_SEQUENCES, SEQUENCE_LENGTH, 2])
            sys.exit()

        # break up sequence data into windowed views of sequence on new dimension 
        shape = (TOTAL_NUMBER_SEQUENCES, DELAY_LENGTH, 2, SEQUENCE_LENGTH - DELAY_LENGTH + 1)
        strides = (*data_delay.strides, 2*data_delay.itemsize)
        data = as_strided(data_delay, shape, strides)
    
        try:
            assert((data[0,:,:,0] == data_delay[0,:DELAY_LENGTH,:]).all())
        except:
            print("Test 3 failed", data[0,:,:,0], data[0,:,:,-1], data_delay[0,:DELAY_LENGTH + 1,:], data_delay[0,-DELAY_LENGTH-1:,:])
            sys.exit()

        np.savez(open(DATA_FILE + ".npz", "wb"), data=data, target=target)

    # train!
    print('training model')
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)


        iter_error = []
        for i in range(ITERATIONS):
            batch = np.random.randint(0, int(.95*TOTAL_NUMBER_SEQUENCES), BATCH_SIZE)
            seq_error = []
            for j in range(SEQUENCE_LENGTH-DELAY_LENGTH+1):
                input_batch = data[batch,:,:,j]
                target_batch = target[batch,j:j+1,:]

                feed_dict={sy_input: input_batch, sy_target: target_batch}

                _, error = sess.run([minimize, sy_error], feed_dict)
                seq_error.append(error)
                #    if ((i%PRINTEVERY) == 0): 
                #print("It: %d | test err: %f" % (i, error))
            iter_error.append(seq_error)

        # test!!
        batch = list(range(int(.95*TOTAL_NUMBER_SEQUENCES), TOTAL_NUMBER_SEQUENCES))
        val_error = []
        for j in range(SEQUENCE_LENGTH-DELAY_LENGTH+1):
            test_input = data[batch,:,:,j]
            test_target = target[batch,j:j+1,:]

            feed_dict={sy_input: test_input, sy_target: test_target}
        
            _, error = sess.run([minimize, sy_error], feed_dict)
            val_error.append(error)
            #print("Symbol: %d | validation err: %f" % (j, error))

        # save model
        #saver.save(sess, MODEL_FILE)


    # write configuration
    # with open(DATA_FILE + "_err.txt", "w") as err:
    #     err.write("Delay length: {}\n".format(DELAY_LENGTH))
    # write sequence errors
    with open(DATA_FILE + "_err.txt", "wb") as err:
        np.savetxt(err, iter_error, fmt='%.6f', delimiter=' ')
        np.savetxt(err, val_error, fmt='%.6f', delimiter=' ')
###############################################################################


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
