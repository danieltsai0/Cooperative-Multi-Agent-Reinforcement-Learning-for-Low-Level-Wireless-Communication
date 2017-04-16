############################################################
#
#  Basic Learning Receiver
#  Shane Barratt
#
#  Simulates a learning transmitter with a fixed receiver.
#
############################################################ 


from environment import Environment
from util import *

import tensorflow as tf
import numpy as np
import IPython as ipy
import itertools
import matplotlib.pyplot as plt


class NeuralTransmitter(object):
    def __init__(self, n_bits=4):

        # Static network vars
        n_h1 = 8*n_bits
        n_h2 = 2*n_h1
        self.num_actions = 2**n_bits

        # Placeholders
        self.input = tf.placeholder(shape=[None, n_bits], dtype=tf.float32) # -1 or 1
        self.action_r = tf.placeholder(shape=[None], dtype=tf.float32) # radius in polar coordinates
        self.action_theta = tf.placeholder(shape=[None], dtype=tf.float32) # angle in radians
        self.adv = tf.placeholder(shape=[None], dtype=tf.float32) # advantages for gradient estimation
        self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)

        # Layers
        h1 = tf.contrib.layers.fully_connected(
            inputs = self.input,
            num_outputs = n_h1,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )
        h2 = tf.contrib.layers.fully_connected(
            inputs = self.input,
            num_outputs = n_h2,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )
        # Outputs Layers
        theta = np.pi * tf.contrib.layers.fully_connected(
                inputs = h2,
                num_outputs = 1,
                activation_fn = tf.nn.tanh, # tanh for -1 to 1
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            )

        r_scale = 2*tf.Variable(.707, name='r_scale')
        r = r_scale * tf.contrib.layers.fully_connected (
                inputs = h2,
                num_outputs = 1,
                activation_fn = tf.nn.sigmoid, # sigmoid for 0 to 1
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            )

        # logs of std dev
        r_logstd = tf.get_variable("r_logstd", [self.num_actions], initializer=tf.zeros_initializer)
        theta_logstd = tf.get_variable("theta_logstd", [self.num_actions], initializer=tf.zeros_initializer)

        # randomized actions
        r_distr = tf.contrib.distributions.Normal(r, tf.exp(r_logstd))
        theta_distr = tf.contrib.distributions.Normal(theta, tf.exp(theta_logstd))
        self.action_r_sample = r_distr.sample()
        self.action_theta_sample = theta_distr.sample()

        # for forming surrogate loss
        self.r_logprob = r_distr.log_prob(self.action_r)
        self.theta_logprob = theta_distr.log_prob(self.action_theta)

        surr = - tf.reduce_mean(self.adv * (self.theta_logprob + self.r_logprob))
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def transmit(self, x):

        # run policy
        theta, r = self.sess.run([self.action_theta_sample, self.action_r_sample], feed_dict={
                self.input: x
            })

        theta, r = theta[0][0], r[0][0]
        return r * np.array([np.cos(theta), np.sin(theta)]), r, theta

    def constellation(self):
        """
        Plots a constellation diagram. (https://en.wikipedia.org/wiki/Constellation_diagram)
        """
        bitstrings = list(itertools.product([0, 1], repeat=self.n_bits))

        plt.figure()
        for bs in bitstrings:
            theta, r = self.transmit(np.array(bs)[None])
            plt.scatter(r*np.cos(theta), r*np.sin(theta), label=str(bs))

        plt.legend()
        plt.show()

if __name__ == '__main__':
    # Statics
    n_bits = 2
    steps_per_episode = 1000
    train_iter = 1000
    stepsize = 1e-2

    env = Environment(n_bits=n_bits, l=.001)
    nt = NeuralTransmitter(n_bits=n_bits)

    for i in range(train_iter):

        obs, acs_r, acs_t, rewards = [], [], [], []
        for _ in range(steps_per_episode):
            # tx
            tx_inp = env.get_input_transmitter()
            tx_out, r, theta = nt.transmit(tx_inp[None])
            env.output_transmitter(tx_out)

            # rx
            rx_inp = env.get_input_receiver()
            rx_out, dist = None, float("inf")
            for k in psk.keys():
                d = np.linalg.norm(rx_inp - psk[k], ord=2)
                if d < dist:
                    rx_out = np.array(k)
                    dist = d
            env.output_receiver(rx_out)

            # rewards
            tx_reward = env.reward_transmitter()
            rx_reward = env.reward_receiver()

            obs.append(tx_inp)
            acs_r.append(r)
            acs_t.append(theta)
            rewards.append(tx_reward)

        if i > 300:
            stepsize = 1e-3
        if i > 600:
            stepsize = 1e-4

        _ = nt.sess.run(nt.update_op, feed_dict={nt.input:obs, nt.action_r:acs_r, nt.action_theta:acs_t, nt.adv:rewards, nt.stepsize:stepsize})
        if i % 50 == 0:
            print("iteration number",i)
            print("avg reward for this episode:",np.average(rewards))
            fn = fn_base+str(i)+".png"
            nr.draw_boundaries("",fn)

    nt.constellation()
