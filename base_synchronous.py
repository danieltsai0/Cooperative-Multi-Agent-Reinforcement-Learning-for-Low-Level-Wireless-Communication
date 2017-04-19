############################################################
#
# Synchronous learning of both Rx and Tx. 
#
############################################################ 

from base_transmitter import NeuralTransmitter
from base_receiver import NeuralReceiver
from environment import Environment
from util import *

import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import math

# Modulation statics
n_bits = 4
n_input = 1
mod_scheme, legend_map, inv_legend_map = get_mod_stuff(n_bits)
# Learning statics
steps_per_episode = 128
train_iter = 2000
fn_base = "images/decision_iter_%d.png"
title = "Decision Boundaries over Complex Plane for Fixed Tx and \nLearning Rx, fixed and known number of transmitted bits.\nIteration number %d."
tx_stepsize = 1e-2
rx_stepsize = 1e-2


# define environment and neural Rx
env = Environment(n_bits=n_bits, l=.05, noise=lambda x: x + np.random.normal(loc=0.0, scale=.3, size=2))
nt = NeuralTransmitter(n_bits=n_bits, steps_per_episode=steps_per_episode)
nr = NeuralReceiver(n_bits=n_bits, n_input=n_input)

# Train synchronously
for i in range(train_iter):

    # Tx accum
    tx_obs = np.empty((0, n_bits))
    tx_acs_re = np.empty(0)
    tx_acs_im = np.empty(0)
    tx_adv = np.empty(0)
    # Rx accum
    rx_obs = np.empty((0, 2*n_input))
    rx_acs = np.empty(0)
    rx_adv = np.empty(0)
    for _ in range(steps_per_episode):
        # tx
        tx_inp = zero_to_neg_one(env.get_input_transmitter())
        tx_out, re, im = nt.transmit(tx_inp[None])
        env.output_transmitter(tx_out)

        # rx
        rx_inp = env.get_input_receiver()
        rx_out, out = nr.output(rx_inp[None])
        env.output_receiver(rx_out) # give env the cartesian representation of action

        # rewards
        tx_reward = env.reward_transmitter()
        rx_reward = env.reward_receiver()

        # Tx accum
        tx_obs = np.r_[tx_obs, tx_inp[None]]
        tx_acs_re = np.r_[tx_acs_re, re]
        tx_acs_im = np.r_[tx_acs_im, im]
        tx_adv = np.r_[tx_adv, tx_reward]
        # Rx accum
        rx_obs = np.r_[rx_obs, rx_inp[None]]
        rx_acs = np.r_[rx_acs, out]
        rx_adv = np.r_[rx_adv, rx_reward]

    if i > 300:
        stepsize = 5e-3

    _ = nt.sess.run(nt.update_op, 
                    feed_dict={nt.input:tx_obs, nt.actions_re:tx_acs_re, nt.actions_im:tx_acs_im, 
                               nt.adv:tx_adv, nt.stepsize:tx_stepsize})
    _ = nr.sess.run(nr.update_op, 
                    feed_dict={nr.input:rx_obs, nr.actions:rx_acs, nr.adv:rx_adv, nr.stepsize:rx_stepsize})

    if i % 10 == 0:
        print("iteration number",i)
        print("avg reward for this episode:",np.average(rx_adv))
        nt.constellation(iteration=i, groundtruth=mod_scheme)
        nr.draw_boundaries(title, fn_base, i, legend_map, inv_legend_map)