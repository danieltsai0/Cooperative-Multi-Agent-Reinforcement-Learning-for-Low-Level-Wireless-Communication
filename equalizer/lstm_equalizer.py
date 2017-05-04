import tensorflow as tf
import numpy as np
import scipy.signal as signal
import sys
import getopt
import plot_equalizer
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

class equalizer():
    def __init__(self, 
                 CONSTELLATION='8psk',
                 DELAY_LENGTH=5,
                 TOTAL_NUMBER_SEQUENCES=1000,
                 BATCH_SIZE=50, 
                 HIDDEN_DIM=20,
                 NUM_LAYERS=2,
                 ITERATIONS=400,
                 LEARNING_RATE=1e-6):
        
        #######################################################################
        # CONFIGURATION
        self.ITERATIONS = ITERATIONS
        self.LEARNING_RATE = LEARNING_RATE

        self.SEQUENCE_LENGTH = 100 # number of symbols per sequence
        self.TOTAL_NUMBER_SEQUENCES = TOTAL_NUMBER_SEQUENCES # total number of sequences in datapool
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_DIM = HIDDEN_DIM
        self.NUM_LAYERS = NUM_LAYERS # number of layers in the LSTM
        self.OUTPUT_DIM = 2 # defines maximum number of output bits
        self.DELAY_LENGTH = DELAY_LENGTH
        self.CONSTELLATION = CONSTELLATION
        self.DATA_FILE = 'equalizer_results/' + CONSTELLATION + '_' + str(DELAY_LENGTH)
        self.MODEL_FILE = 'equalizer_models/' + CONSTELLATION + '_' + str(DELAY_LENGTH)
        self.PRINTEVERY = 50 # how often the training error is reported
        self.plot = plot_equalizer.plot()


    def run(self):
        data, target = self.build_experiment(self.SEQUENCE_LENGTH,
                                             self.TOTAL_NUMBER_SEQUENCES,
                                             self.CONSTELLATION,
                                             self.DELAY_LENGTH)
        iter_error, val_error = self.build_graph(data, target)
        self.plot_results()
        self.write_results(iter_error, val_error)
      

    ################################################################################
    #
    # COMPUTATIONAL GRAPH
    ###############################################################################
    def build_graph(self, data, target):
        # input dimensions: (#batch, #timestep, data dimension)
        #batch_size = tf.placeholder(tf.int32)
        batch_size = self.BATCH_SIZE
        sy_input = tf.placeholder(tf.float32, shape=[batch_size, self.DELAY_LENGTH, 2]) 
        sy_target = tf.placeholder(tf.float32, shape=[batch_size, 1, 2])

        sy_lstm = tf.contrib.rnn.LSTMCell(self.HIDDEN_DIM)
        sy_stacked_lstm = tf.contrib.rnn.MultiRNNCell([sy_lstm]*self.NUM_LAYERS)
        sy_initial_state = sy_state = sy_stacked_lstm.zero_state(batch_size, dtype=tf.float32)

        # Version 1: don't unroll LSTM
        sy_outputs, sy_state = tf.nn.dynamic_rnn(sy_stacked_lstm, 
                                             sy_input, 
                                             initial_state=sy_initial_state, 
                                             time_major=False)

        # we are only interested in the last output
        sy_last = tf.unstack(sy_outputs, axis=1)[-1] 

        # transform outputs of LSTM to output target (equalized sequence)
        sy_w_real = tf.Variable(tf.truncated_normal([self.HIDDEN_DIM, 1]), name="output_weight_real")
        sy_b_real = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias_real")
        sy_output_real = tf.matmul(sy_last, sy_w_real) + sy_b_real
        sy_w_imag = tf.Variable(tf.truncated_normal([self.HIDDEN_DIM, 1]), name="output_weight_imag")
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
        #
        # COMPUTATIONAL GRAPH
        ###############################################################################
        print('training model')
        #saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.as_default()
            init = tf.global_variables_initializer()
            sess.run(init)

            iter_error = []
            for i in range(self.ITERATIONS):
                batch = np.random.randint(0, int(.95*self.TOTAL_NUMBER_SEQUENCES), self.BATCH_SIZE)
                
                seq_error = []
                for j in range(self.SEQUENCE_LENGTH-self.DELAY_LENGTH+1):
                    input_batch = data[batch,:,:,j]
                    target_batch = target[batch,j:j+1,:]

                    feed_dict={sy_input: input_batch, sy_target: target_batch}

                    _, error = sess.run([minimize, sy_error], feed_dict)
                    seq_error.append(error)
                
                if ((i%self.PRINTEVERY) == 0): 
                    print("It: %d | test err: %f" % (i, error))
                iter_error.append(seq_error)
    
            # test!!
            batch = list(range(int(.95*self.TOTAL_NUMBER_SEQUENCES), self.TOTAL_NUMBER_SEQUENCES))
            val_error, val_output = [], []
            for j in range(self.SEQUENCE_LENGTH-self.DELAY_LENGTH+1):
                test_input = data[batch,:,:,j]
                test_target = target[batch,j:j+1,:]

                feed_dict={sy_input: test_input, sy_target: test_target}
            
                error, output = sess.run([sy_error, sy_output], feed_dict)
                val_error.append(error)
                val_output.append(output)
                #print("Symbol: %d | validation err: %f" % (j, error))

            self.plot.set_output(np.hstack(val_output))
            
            # save model
            #saver.save(sess, MODEL_FILE)
        return iter_error, [val_error]
        
    ###########################################################################
    # EXPERIMENT 
    def build_experiment(self, 
                         SEQUENCE_LENGTH, 
                         TOTAL_NUMBER_SEQUENCES, 
                         CONSTELLATION,
                         DELAY_LENGTH):
        # generate data 
        try:
            npz = np.load(open(self.DATA_FILE + ".npz", "rb"))
            data, target = npz['data'], npz['target']
            # self.plot.set_channel(DELAY)
            # self.plot.set_transmitted(data_delay[int(TOTAL_NUMBER_SEQUENCES*.95):, :, :])
            # self.plot.set_target(target[int(TOTAL_NUMBER_SEQUENCES*.95):, :, :])
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

            # create training data
            DELAY = np.random.normal(0, 1/np.sqrt(2), (2, DELAY_LENGTH))
            DELAY = np.array([[complex(x, y) for (x, y) in zip(DELAY[0,:], DELAY[1,:])]])
            
            try:
                assert(DELAY_LENGTH == DELAY.shape[1])
            except:
                print(DELAY.shape, DELAY.size)
                sys.exit()

            data_delay = signal.convolve2d(DELAY, data)[:, DELAY_LENGTH-1:]

            try: assert(data_delay.shape[1] == SEQUENCE_LENGTH)
            except: print("Test 1 failed", data_delay.shape); sys.exit()
    
            data_delay = np.stack([data_delay.real, data_delay.imag], 2) 

            try: 
                assert(data_delay.shape == (TOTAL_NUMBER_SEQUENCES, SEQUENCE_LENGTH, 2))
            except:
                print("Test 2 failed", data_delay.shape, [TOTAL_NUMBER_SEQUENCES, SEQUENCE_LENGTH, 2])
                sys.exit()

            # store channel and validation set for plotting
            self.plot.set_channel(DELAY)
            self.plot.set_transmitted(data_delay[int(TOTAL_NUMBER_SEQUENCES*.95):, :, :])
            self.plot.set_target(target[int(TOTAL_NUMBER_SEQUENCES*.95):, :, :])
            

            # break up sequence data into windowed views of sequence on new dimension 
            shape = (TOTAL_NUMBER_SEQUENCES, DELAY_LENGTH, 2, SEQUENCE_LENGTH - DELAY_LENGTH + 1)
            print(data_delay.strides)
            strides = (*data_delay.strides, 2*data_delay.itemsize)
            data = as_strided(data_delay, shape, strides)
    
            try:
                assert((data[0,:,:,0] == data_delay[0,:DELAY_LENGTH,:]).all())
            except:
                print("Test 3 failed", data[0,:,:,0], data[0,:,:,-1], data_delay[0,:DELAY_LENGTH + 1,:], data_delay[0,-DELAY_LENGTH-1:,:])
                sys.exit()

            np.savez(open(self.DATA_FILE + ".npz", "wb"), data=data, target=target)
        return data, target
                    
    def write_results(self, iter_error, val_error):
        # write configuration
        # with open(DATA_FILE + "_err.txt", "w") as err:
        #     err.write("Delay length: {}\n".format(DELAY_LENGTH))
        # write sequence errors
        print("writing results")
        with open(self.DATA_FILE + "_err.txt", "wb") as err:
            np.savetxt(err, iter_error, fmt='%.6f', delimiter=' ')
            np.savetxt(err, val_error, fmt='%.6f', delimiter=' ')

    def plot_results(self):
        print('plotting results')
        self.plot.plot_val(self.DATA_FILE + '.png')

def usage():
    print("c : constellation type\n" + 
          "d : delay length\n" + 
          "s : total number of sequences (not implemented)\n" +
          "b : batch size\n" +
          "l : hidden dimension size\n" + 
          "i : training iterations\n" +
          "r : learning rate\n" + 
          "f : parameter configuration file name (overrides other args)")
    
if __name__ == "__main__":
    CONSTELLATION='8psk'
    DELAY_LENGTH=5
    TOTAL_NUMBER_SEQUENCES=1000
    BATCH_SIZE=50
    HIDDEN_DIM=20
    NUM_LAYERS=2
    ITERATIONS=400
    LEARNING_RATE=1e-6

    if len(sys.argv) > 1: # has arguments
        # parse flags
        try:
            opts, args = getopt.getopt(sys.argv, "c:d:s:b:l:i:r:f:")
        except:
            usage()
            sys.exit(-1)

        args = args[1:]
        args.reverse()
        while(args):
            opt = args.pop()
            if opt == '-c':
                CONSTELLATION=args.pop()
            if opt == '-d':
                DELAY_LENGTH=int(args.pop())
            # if opt == '-s':
            #     TOTAL_NUMBER_SEQUENCES=arg
            if opt == '-b':
                BATCH_SIZE=int(args.pop())
            if opt == '-l':
                NUM_LAYERS=int(args.pop())
            if opt == '-i':
                ITERATIONS=int(args.pop())
            if opt == '-r':
                LEARNING_RATE=float(args.pop())

    # create and run model
    equal = equalizer(CONSTELLATION,
                      DELAY_LENGTH,
                      TOTAL_NUMBER_SEQUENCES,
                      BATCH_SIZE, 
                      HIDDEN_DIM,
                      NUM_LAYERS,
                      ITERATIONS,
                      LEARNING_RATE)
    equal.run()
