############################################################
#
#  Basic Learning Transmitter
#  Shane Barratt <stbarratt@gmail.com>
#
#  Network takes in a n_bit long sequence of bits and outputs 
#  a continuous distribution over x and y, which denote the
#  two axes in the complex plane.
#
############################################################ 

from util import *

import tensorflow as tf
import numpy as np
import IPython as ipy
import itertools
import matplotlib.pyplot as plt
import time

# normalized constant initializer from cs 294-112 code
def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class NeuralTransmitter(object):
    def __init__(self, n_bits, n_hidden, stepsize, l, groundtruth, uid):
        # Network parameters
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.stepsize = stepsize
        self.l = l # loss rate for power 
        # Misc. parameters
        self.groundtruth = groundtruth
        self.im_dir = 'figures/'+str(uid)+'/'
        create_dir(self.im_dir)
        self.im_dir += '%d.png'

        # Network
        self.input = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.actions_x = tf.placeholder(tf.float32, [None]) # radius in polar coordinates
        self.actions_y = tf.placeholder(tf.float32, [None]) # angle in radians
        self.adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        # self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)
# 
        # Hidden Layer
        self.h1 = tf.contrib.layers.fully_connected(
            inputs = self.input,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )

        self.h2 = tf.contrib.layers.fully_connected(
            inputs = self.h1,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )

        # Outputs
        self.x_mean = tf.squeeze(tf.contrib.layers.fully_connected (
                inputs = self.h2,
                num_outputs = 1,
                activation_fn = None,
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(1.5)
            ))
        self.y_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs = self.h2,
                num_outputs = 1,
                activation_fn = None, 
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.0)
            ))
        self.x_logstd = tf.Variable(0.)
        self.y_logstd = tf.Variable(0.)
        self.x_std = tf.exp(self.x_logstd)
        self.y_std = tf.exp(self.y_logstd)

        # randomized actions
        self.x_distr = tf.contrib.distributions.Normal(self.x_mean, self.x_std)
        self.y_distr = tf.contrib.distributions.Normal(self.y_mean, self.y_std)

        self.x_sample = self.x_distr.sample()
        self.y_sample = self.y_distr.sample()

        # for forming surrogate loss
        self.x_logprob = self.x_distr.log_prob(self.actions_x)
        self.y_logprob = self.y_distr.log_prob(self.actions_y)

        self.surr = - tf.reduce_mean(self.adv * (self.y_logprob + self.x_logprob))
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    """
    Updates the transmitter based on the input, x and y SAMPLE outputs
    and the advantage (which is just the loss right now).
    """
    def policy_update(self):

        self.sess.run([self.update_op], feed_dict={
                self.input: self.trans_input,
                self.actions_x: self.x_accum,
                self.actions_y: self.y_accum,
                self.adv: self.adv_accum,
                # self.stepsize: self.stepsize
            })

    """
    Wrapper function for policy update. Receives the bit format of the guess of the 
    preamble guess and computes a loss over it.

    Inputs: 
        preamble_g_g_bit: bit format of the guess of the preamble guess - taken from
                          this Actor's receiver unit
    """
    # Receive reward signal from other agent. Should be of same length as actions
    def update(self, preamble_g_g_bit):
        self.adv_accum = - self.ridge_loss(preamble_g_g_bit)
        # print("adv_accum.shape:",self.adv_accum.shape)  
        print("avg_reward:",np.average(self.adv_accum))  
        self.policy_update()

    
    """
    Transmits the input signal as a sequence of complex numbers.

    Inputs: 
        signal: bit format signal

    Outputs: 
        modulated output: complex signal to be transmitted
    """
    def transmit(self, signal):
        # get chunk of data to transmit
        self.trans_input = signal
        # print("trans_input.shape:",self.trans_input.shape)
        # run policy
        x, y = self.sess.run([self.x_sample, self.y_sample], feed_dict={
                self.input: self.trans_input
            })
        # store actions
        self.x_accum = np.array(x)
        self.y_accum = np.array(y)   
        # print("x_accum.shape:",self.x_accum.shape) 
        # print("y_accum.shape:",self.y_accum.shape)        
        self.trans_output = np.array([x,y]).T
        # print("trans_output.shape:",self.trans_output.shape)
        return self.trans_output

    """
    Evaluates the means of the distribution for plotting purposes.

    Inputs: 
        data: bitwise array to be modulated
    """
    def evaluate(self, data):
        # run policy
        x, y = self.sess.run([self.x_mean, self.y_mean], feed_dict={
                self.input: data
            })     
        return np.array([x,y]).T

    """
    Visualize the centroids of the distributions of the network.

    Inputs: 
        iteration: used for plotting purposes
    """
    def visualize(self, iteration):
        """
        Plots a constellation diagram. (https://en.wikipedia.org/wiki/Constellation_diagram)
        """
        bitstrings = list(itertools.product([0, 1], repeat=self.n_bits))

        plt.figure(figsize=(4, 4))
        for bs in bitstrings:
            x,y = self.evaluate(np.array(bs)[None])
            plt.scatter(x, y, label=str(bs))
            plt.annotate(str(bs), (x, y), size=5)
        plt.axvline(0)
        plt.axhline(0)
        plt.xlim([-3., 3.])
        plt.ylim([-3., 3.])

        if self.groundtruth:
            for k in self.groundtruth.keys():
                x_gt, y_gt = self.groundtruth[k]
                plt.scatter(x_gt, y_gt, s=5, color='purple')
                plt.annotate(''.join([str(b) for b in k]), (x_gt, y_gt), size=5)

        plt.savefig(self.im_dir % iteration)
        plt.close()



    ##################
    # Loss functions #
    ##################

    """
    L1 loss

    Inputs:
        signal: bit format signal to be compared to the original input
    """
    def lasso_loss(self, signal):
        # print("trans_output.shape:",self.trans_output.shape)
        # print("trans_output_sum.shape:",(self.l*np.sum(self.trans_output**2,axis=1)).shape)
        # print("trans_input.shape:",self.trans_input.shape)
        # print("signal.shape:",signal.shape)
        # print("linalg.shape:",np.linalg.norm(self.trans_input - signal, ord=1,axis=1).shape)
        return np.linalg.norm(self.trans_input - signal, ord=1, axis=1) + self.l*np.sum(self.trans_output**2,axis=1)

    """
    L2 loss

    Inputs:
        signal: bit format signal to be compared to the original input
    """
    def ridge_loss(self, signal):
        return np.linalg.norm(self.trans_input - signal, axis=1) + self.l*np.sum(self.trans_output**2,axis=1)