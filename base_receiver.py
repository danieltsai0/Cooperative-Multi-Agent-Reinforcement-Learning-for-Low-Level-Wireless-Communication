############################################################
#
#  Basic Learning Receiver
#  Daniel Tsai <daniel_tsai@berkeley.edu>
#
#  Simulates a learning receiver with a fixed transmitter.
#
#  Assumptions:
#   - 'k' is known (i.e. the number of bits that the tramsitter
#       is trying to encode in every symbol)
#
############################################################ 


from environment import Environment
from util import *

import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import math


############################################################
# Function estimatior for decoding transmission.
# Assumes fixed 'k' (n_bits). Input is assumed to be a single
# complex number in Cartesian coordinates. 
###########################################################

class NeuralReceiver:
    #####
    # Inputs:
    #   n_bits: size of output
    #   n_input: size of input (number of complex numbers)
    #   steps_per_epsiode: number of transmissions per episode
    #   step_size: learning rate
    #   state: 
    #       if 0: use epsilon greedy
    #       if 1: use boltzmann exploration
    #
    #
    # Placeholders:
        # input: matrix where each row is an array of complex numbers (x,y,x,y...) etc
        #     shape: (?,2*n_input)
        # actions: integers representing the network's guess for cartesian coordiantes (see util function for coord_to_int)
        #     shape: (?,) 
        # adv: reward for each action taken
        #     shape: (?,)

    def __init__(self, n_bits=2, n_input=1):

        # Static network vars
        n_h1 = 16*n_input
        n_h2 = n_h1//2
        self.num_actions = 2**n_bits
        self.n_bits = n_bits
        self.epsilon = 0.05

        # Placeholders
        self.input = tf.placeholder(shape=[None,2*n_input], name="input", dtype=tf.float32) # NOTE: second dimension is twice of 'k' 
        self.actions = tf.placeholder(shape=[None], name="output", dtype=tf.int32)
        self.adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)

        # Layers
        h1 = tf.contrib.layers.fully_connected(
            inputs = self.input,
            num_outputs = n_h1,
            activation_fn = tf.nn.relu, 
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )
        h2 = tf.contrib.layers.fully_connected(
            inputs = h1,
            num_outputs = n_h2,
            activation_fn = tf.nn.relu, 
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )
        self.logits = tf.contrib.layers.fully_connected(
            inputs = h2,
            num_outputs = self.num_actions,
            activation_fn = tf.nn.log_softmax, 
            weights_initializer = normc_initializer(0.1),
            biases_initializer = tf.constant_initializer(.1)
        )

        # Other tensors
        self.logprobs = tf.nn.log_softmax(self.logits)
        self.sampled_act = categorical_sample_logits(self.logits)
        num_samples = tf.shape(self.input)[0]

        # The logprobs of the actions taken
        self.selected_logprobs = fancy_slice_2d(self.logprobs, tf.range(num_samples), self.actions)

        # Define loss and optimizer
        self.surr = -tf.reduce_mean(self.adv * self.selected_logprobs)
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr)

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # returns output in format of (key to mod scheme map, output)
    def output(self, rx_inp):
        out = self.sess.run(self.sampled_act, feed_dict={self.input:rx_inp})[0]
        return int_to_coord(out, self.n_bits), out

    def epsilon_greedy(self, actions, epsilon):
        # return optimal action
        if np.random.rand() > epsilon:
          return action
        # return random action (i.e. either 0 or 1)
        return tf.constant(np.random.randint(0,self.num_actions))

    # Not implemented yet
    def boltzmann_exploration(self, action):
        return action     


    def draw_boundaries(self, title, fn, iteration):
        # Useful maps
        color_map = {
                    0: 'deepskyblue',
                    1: 'orangered',
                    2: 'fuchsia',
                    3: 'lime'
                }
        legend_map = {
                    0: '(0,0)',
                    1: '(0,1)',
                    2: '(1,0)',
                    3: '(1,1)'
                }
        inv_legend_map = {v: k for k, v in legend_map.items()}

        # Generate graph points
        points = np.mgrid[-1.5:1.5:.05, -1.5:1.5:.05].transpose(1,2,0)
        points = np.reshape(points, (points.shape[0]*points.shape[1], points.shape[2])) # squeeze out extra dim
        # Get actions
        actions = self.sess.run(self.sampled_act, feed_dict={self.input:points})

        # Plot decisions, each element is (x,y,action)
        elems = [None]*self.num_actions
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_axis_bgcolor('black')

        for e in np.c_[points,actions]:
            elems[int(e[2])] = plt.plot(e[0], e[1], 'o', color=color_map[e[2]], label=legend_map[e[2]], ms=6, mew=2)[0]

        # plt.legend()
        leg = ax.legend(handles=elems)

        for text in leg.get_texts():
            text.set_color(color_map[inv_legend_map[text.get_text()]])

        plt.title(title % iteration)
        plt.xlabel("Real axis")
        plt.ylabel("Imaginary axis")
        plt.savefig('figures/Rx/%d.png' % iteration)
        plt.close()





############################################################
# Run basic simulation in which a fixed transmitter tries to
# send 4 bits of information via QPSK. Learning Receiver 
# attempts to learn this transmission protocol.
###########################################################

if __name__ == '__main__':

    # Statics
    n_bits = 2
    steps_per_episode = 1000
    train_iter = 1000
    fn_base = "images/decision_iter_%d.png"
    title = "Decision Boundaries over Complex Plane for Fixed Tx and \nLearning Rx, fixed and known number of transmitted bits.\nIteration number %d."
    stepsize = 1e-2

    # define environment and neural Rx
    env = Environment(n_bits=2, l=.001, noise=lambda x: x + np.random.normal(loc=0.0, scale=.3, size=2))
    nr = NeuralReceiver(n_bits=2, n_input=1)

    # Train Receiver
    for i in range(train_iter):

        obs, acs, rewards = [], [], []
        for _ in range(steps_per_episode):
            # tx
            tx_inp = env.get_input_transmitter()
            tx_out = psk.get(tuple(tx_inp))
            env.output_transmitter(tx_out)

            # rx
            rx_inp = env.get_input_receiver()
            rx_out, out = nr.sess.run(nr.sampled_act, feed_dict={nr.input:rx_inp[None]})[0]
            env.output_receiver(rx_out) # give env the cartesian representation of action

            # rewards
            tx_reward = env.reward_transmitter()
            rx_reward = env.reward_receiver()

            obs.append(rx_inp)
            acs.append(rx_out)
            rewards.append(rx_reward)

        obs = np.array(obs)
        acs = np.array(out)
        rewards = np.array(rewards)

        if i > 300:
            stepsize = 5e-3

        _ = nr.sess.run(nr.update_op, feed_dict={nr.input:obs, nr.actions:acs, nr.adv:rewards, nr.stepsize:stepsize})
        if i % 10 == 0:
            print("iteration number",i)
            print("avg reward for this episode:",np.average(rewards))
            nr.draw_boundaries(title, fn_base, i)

    nr.draw_boundaries(title, fn_base, i)