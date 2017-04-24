"""
shane-lstm.py
Author: Shane Barratt

Neural Transmitter that takes variable input and produces variable output.
"""

import numpy as np
import tensorflow as tf
import IPython as ipy

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class NeuralTransmitterLSTM(object):
    """
    Neural transmitter that provides an end-to-variable length bitstrings and produces variable length complex modulations.

    LSTM 1 (the protocol) computes an R^d hidden representation of a given bitstring input:
        p(h|x; \theta)
    LSTM 2 (the decoder) computes the probability of outputting a complex modulation given the previous modulations and the hidden representation:
        p(c_i|c_0, ..., c_{i-1}, h; \theta)
    """
    def __init__(self,
            num_hidden_1=25,
            num_hidden_2=25,
            max_length_1=100,
            max_length_2=100):
        # Parameters
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.max_length_1 = max_length_1
        self.max_length_2 = max_length_2

        # LSTM "Encoder"
        with tf.variable_scope('lstm1'):
            # placeholders
            self.sy_x_1 = tf.placeholder(tf.float32, [None, self.max_length_1, 1]) # batch of bitstring inputs
            self.sequence_lengths_1 = tf.placeholder(tf.int32, [None]) # lengths of each bitstring input in the batch

            # network
            self.cell_1 = tf.contrib.rnn.LSTMCell(self.num_hidden_1, state_is_tuple=True)
            self.outputs_encoder, _ = tf.nn.dynamic_rnn(self.cell_1, self.sy_x_1, dtype=tf.float32, sequence_length=self.sequence_lengths_1)
            self.outputs_encoder = tf.transpose(self.outputs_encoder, [1, 0, 2])
            self.h = tf.gather(self.outputs_encoder, int(self.outputs_encoder.get_shape()[0]) - 1) # get the last hidden state of the LSTM

        # LSTM "Decoder"
        # each unrolling of the LSTM receives the output of the previous cell as input, and produces, with the hidden state, the probability of the next output
        with tf.variable_scope('lstm2'):
            # placeholders
            self.sy_x_2 = tf.placeholder(tf.float32, [None, self.max_length_2, 2]) # Inputs
            self.sy_y_2 = tf.placeholder(tf.float32, [None, self.max_length_2, 2])
            self.sequence_lengths_2 = tf.placeholder(tf.int32, [None])
            self.sy_batch_size = tf.placeholder(tf.int32, []) # batch size for computing the "zero" initial state
            self.c_state = tf.placeholder(tf.float32, [None, self.num_hidden_2])
            self.h_state = tf.placeholder(tf.float32, [None, self.num_hidden_2])
            self.sy_initial_state = tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state) # initial state for unrolling

            # network
            self.cell_2 = tf.contrib.rnn.LSTMCell(self.num_hidden_2, state_is_tuple=True)
            self.zero_initial_state = self.cell_2.zero_state(self.sy_batch_size, tf.float32)
            self.outputs_decoder, self.states_decoder = tf.nn.dynamic_rnn(self.cell_2, self.sy_x_2, dtype=tf.float32, sequence_length=self.sequence_lengths_2, initial_state=self.sy_initial_state)

            # weights for the mean (x, y) output layer
            self.W_out = tf.Variable(normc_initializer(1.0)(
                [self.num_hidden_2 + self.num_hidden_1, 2]))
            self.b_out = tf.Variable(tf.constant_initializer(0.0)((2,)))

            # weights for the P(end) output layer
            self.W_endprob = tf.Variable(normc_initializer(1.0)(
                [self.num_hidden_2 + self.num_hidden_1, 1]))
            self.b_endprob = tf.Variable(tf.constant_initializer(0.0)((1,)))

            self.output_projection = lambda x: tf.matmul(tf.concat([x, self.h], axis=-1), self.W_out) + self.b_out
            self.output_endprob_projection = lambda x: tf.matmul([x, self.h], axis=-1), self.W_endprob) + self.b_endprob

            self.outputs_decoder = tf.transpose(self.outputs_decoder, [1, 0, 2])
            self.means = tf.map_fn(self.output_projection, self.outputs_decoder)
            self.endprobs = tf.map_fn(self.output_endprob_projection, self.outputs_decoder)
            self.means = tf.transpose(self.means, [1, 0, 2])
            self.endprobs = tf.transpose(self.endprobs, [1, 0, 2])

            self.logstd = tf.Variable(-1.) # logstd of output positions

            self.distr = tf.contrib.distributions.Normal(self.means, self.logstd)

            self.sy_action_sample = self.distr.sample()

            # TODO: compute log-probability of sequence, consider weighting by length

            # TODO: compute surrogate loss

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def transmit(self, x):
        # consider beam search
        raise NotImplementedError

    def policy_update(self):
        raise NotImplementedError

if __name__ == '__main__':
    nt = NeuralTransmitterLSTM()

    ipy.embed()