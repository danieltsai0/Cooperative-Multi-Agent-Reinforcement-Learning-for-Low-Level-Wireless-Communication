############################################################
#
# WORKING CARTESIAN
#
############################################################ 


from environment import Environment
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
    def __init__(self, n_bits=2, n_hidden=16, steps_per_episode=32, stepsize=1e-3):
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.steps_per_episode = steps_per_episode
        self.stepsize = stepsize

        # Network
        self.input = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.actions_re = tf.placeholder(tf.float32, [None]) # radius in polar coordinates
        self.actions_im = tf.placeholder(tf.float32, [None]) # angle in radians
        self.adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)

        # Hidden Layer
        self.h1 = tf.contrib.layers.fully_connected(
            inputs = self.input,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )

        # Outputs
        self.re_mean = tf.squeeze(tf.contrib.layers.fully_connected (
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = None,
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(.7)
            ))
        self.im_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = None,
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            ))
        self.re_logstd = tf.Variable(-.5)
        self.im_logstd = tf.Variable(-.5)
        self.re_std = tf.exp(self.re_logstd)
        self.im_std = tf.exp(self.im_logstd)

        # randomized actions
        self.re_distr = tf.contrib.distributions.Normal(self.re_mean, tf.exp(self.re_logstd))
        self.im_distr = tf.contrib.distributions.Normal(self.im_mean, tf.exp(self.im_logstd))

        self.sy_re_sample = self.re_distr.sample()
        self.sy_im_sample = self.im_distr.sample()

        # for forming surrogate loss
        self.re_logprob = self.re_distr.log_prob(self.actions_re)
        self.im_logprob = self.im_distr.log_prob(self.actions_im)
        # self.re_logprob = -.5*tf.log(2*np.pi*self.re_std**2) - (self.sy_r - self.re_mean)**2/(2*self.re_std**2)
        # self.im_logprob = -.5*tf.log(2*np.pi*self.im_std**2) - (self.sy_im - self.im_mean)**2/(2*self.im_std**2)

        self.sy_surr = - tf.reduce_mean(self.adv * (self.im_logprob + self.re_logprob))
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.sy_surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def transmit(self, x, evaluate=False):

        # run policy
        if evaluate:
            im, re = self.sess.run([self.im_mean, self.re_mean], feed_dict={
                    self.input: x
            })
            return np.array([re, im])
        else:
            im, re = self.sess.run([self.sy_im_sample, self.sy_re_sample], feed_dict={
                    self.input: x
                })
            return np.array([re, im]), re, im


    def constellation(self, iteration=0, groundtruth=None):
        """
        Plots a constellation diagram. (https://en.wikipedia.org/wiki/Constellation_diagram)
        """
        bitstrings = list(itertools.product([0, 1], repeat=self.n_bits))

        plt.figure(figsize=(4, 4))
        for bs in bitstrings:
            x,y = self.transmit(np.array(bs)[None], evaluate=True)
            plt.scatter(x, y, label=str(bs))
            plt.annotate(str(bs), (x, y), size=5)
        plt.axvline(0)
        plt.axhline(0)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])

        if groundtruth:
            for k in groundtruth.keys():
                x_gt, y_gt = groundtruth[k]
                plt.scatter(x_gt, y_gt, label=str(bs), s=5, color='purple')
                plt.annotate(str(k), (x_gt, y_gt), size=5)

        plt.savefig('figures/Tx/%d.png' % iteration)
        plt.close()

if __name__ == '__main__':
    # set random seeds
    tf.set_random_seed(0)
    np.random.seed(0)

    n_bits = 2
    mod_scheme = psk
    l = .1
    steps_per_episode = 512
    stepsize = 1e-2

    env = Environment(n_bits=n_bits, l=l)

    def rx_decode(rx_inp, m):
        rx_out, dist = None, float("inf")
        for k in m.keys():
            d = np.linalg.norm(rx_inp - m[k], ord=2)
            if d < dist:
                rx_out = np.array(k)
                dist = d
        return rx_out

    nt = NeuralTransmitter(n_bits=n_bits, steps_per_episode=steps_per_episode)

    for i in range(1000):
        rew_pere_ep = 0.0
        xs_accum = np.empty((0, n_bits))
        re_accum = np.empty(0)
        im_accum = np.empty(0)
        adv_accum = np.empty(0)

        for _ in range(steps_per_episode):
            # tx
            tx_inp = zero_to_neg_one(env.get_input_transmitter())
            tx_out, r, im = nt.transmit(tx_inp[None])
            xs_accum = np.r_[xs_accum, tx_inp[None]]
            re_accum = np.r_[re_accum, np.array(r)[None]]
            im_accum = np.r_[im_accum, np.array(im)[None]]
            env.output_transmitter(tx_out)

            # rx
            rx_inp = env.get_input_receiver()
            rx_out = rx_decode(rx_inp, mod_scheme)
            env.output_receiver(rx_out)

            # rewards
            tx_reward = env.reward_transmitter()
            rx_reward = env.reward_receiver()

            adv_accum = np.r_[adv_accum, tx_reward]

        if i > 250:
            stepsize = 1e-3


        nt.sess.run([nt.update_op], feed_dict={
                nt.input: xs_accum,
                nt.actions_re: re_accum,
                nt.actions_im: im_accum,
                nt.adv: adv_accum,
                nt.stepsize: stepsize
            })

        if i % 10 == 0:
            print(np.average(adv_accum))
            nt.constellation(iteration=i, groundtruth=mod_scheme)
 