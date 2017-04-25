############################################################
#
#  Basic Learning Transmitter
#  Shane Barratt <stbarratt@gmail.com>
#
#  Simulates a learning transmitter with a fixed receiver.
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
    def __init__(self, preamble, size_of_episode, n_bits, n_hidden, stepsize, l, groundtruth):
        # Network parameters
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.stepsize = stepsize
        self.l = l
        # Misc. parameters
        self.preamble = preamble
        self.groundtruth = groundtruth
        # values for keeping track of where we are in the preamble
        self.size_of_episode = size_of_episode
        self.index = 0
        self.max_index = (self.preamble.shape[0] // size_of_episode) - 1

        # Network
        self.input = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.actions_r = tf.placeholder(tf.float32, [None]) # radius in polar coordinates
        self.actions_theta = tf.placeholder(tf.float32, [None]) # angle in radians
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

        # Outputs
        self.r_mean = tf.squeeze(tf.contrib.layers.fully_connected (
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = None,
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(1.5)
            ))
        self.theta_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = None, # tanh for -1 to 1
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.0)
            ))
        self.r_logstd = tf.Variable(-1.)
        self.theta_logstd = tf.Variable(0.)
        self.r_std = tf.exp(self.r_logstd)
        self.theta_std = tf.exp(self.theta_logstd)

        # randomized actions
        self.r_distr = tf.contrib.distributions.Normal(self.r_mean, tf.exp(self.r_logstd))
        self.theta_distr = tf.contrib.distributions.Normal(self.theta_mean, tf.exp(self.theta_logstd))

        self.r_sample = self.r_distr.sample()
        self.theta_sample = self.theta_distr.sample()

        # for forming surrogate loss
        self.r_logprob = self.r_distr.log_prob(self.actions_r)
        self.theta_logprob = self.theta_distr.log_prob(self.actions_theta)

        self.surr = - tf.reduce_mean(self.adv * (self.theta_logprob + self.r_logprob))
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def policy_update(self):

        self.sess.run([self.update_op], feed_dict={
                self.input: self.trans_input,
                self.actions_r: self.r_accum,
                self.actions_theta: self.theta_accum,
                self.adv: self.adv_accum,
                # self.stepsize: self.stepsize
            })

    # Receive reward signal from other agent. Should be of same length as actions
    def receive(self, signal):
        self.adv_accum = - self.generic_loss(signal)
        print("adv_accum.shape:",self.adv_accum.shape)  
        print("avg_reward:",np.average(self.adv_accum))  
        self.policy_update()

    # Transmit a signal
    def transmit(self):
        # get chunk of data to transmit
        self.trans_input = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        print("trans_input.shape:",self.trans_input.shape)
        print("transmitter index:",self.index)
        self.index += 1 % self.max_index
        # run policy
        theta, r = self.sess.run([self.theta_sample, self.r_sample], feed_dict={
                self.input: self.trans_input
            })
        # store actions
        self.r_accum = np.array(r)
        self.theta_accum = np.array(theta)   
        print("r_accum.shape:",self.r_accum.shape) 
        print("theta_accum.shape:",self.theta_accum.shape)        
        self.trans_output = (r * np.array([np.cos(theta), np.sin(theta)])).T
        return self.trans_output

    def evaluate(self, data):
        # run policy
        theta, r = self.sess.run([self.theta_mean, self.r_mean], feed_dict={
                self.input: data
            })     
        
        return (r * np.array([np.cos(theta), np.sin(theta)])).T

    # Visualize the decisions of the network
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

        plt.savefig('figures/%d.png' % iteration)
        plt.close()

    """
    Loss functions
    """
    # Input signal is the labels that the receiver received
    def generic_loss(self, signal):
        print("trans_output.shape:",self.trans_output.shape)
        print("trans_output_sum.shape:",(self.l*np.sum(self.trans_output**2,axis=1)).shape)
        print("trans_input.shape:",self.trans_input.shape)
        print("signal.shape:",signal.shape)
        print("linalg.shape:",np.linalg.norm(self.trans_input - signal, ord=1,axis=1).shape)
        return np.linalg.norm(self.trans_input - signal, ord=1, axis=1) + self.l*np.sum(self.trans_output**2,axis=1)