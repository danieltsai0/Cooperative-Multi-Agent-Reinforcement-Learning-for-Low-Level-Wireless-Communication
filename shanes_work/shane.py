############################################################
#
#  Basic Learning Transmitter
#  Shane Barratt <stbarratt@gmail.com>
#
#  Simulates a learning transmitter with a fixed receiver.
#
############################################################ 


from environment import Environment
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
    def __init__(self, n_bits=2, n_hidden=64, steps_per_episode=32, stepsize=5e-3, polar=False):
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.steps_per_episode = steps_per_episode
        self.stepsize = stepsize
        self.polar = polar

        self.step = 0 # current step

        # saved xs, actions and advantages
        self.reset_accum()

        # Network
        self.sy_x = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.sy_x_actions = tf.placeholder(tf.float32, [None]) # x actions for gradient calculation
        self.sy_y_actions = tf.placeholder(tf.float32, [None]) # y actions for gradient calculation
        self.sy_adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # stepsize for gradient step

        # Hidden Layer
        self.h2 = tf.contrib.layers.fully_connected(
            inputs = self.sy_x,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(.1),
            biases_initializer = tf.constant_initializer(.1)
        )

        self.h1 = tf.contrib.layers.fully_connected(
            inputs = self.h2,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(.1),
            biases_initializer = tf.constant_initializer(.1)
        )

        # Outputs
        if self.polar:
            self.r_mean = tf.squeeze(tf.contrib.layers.fully_connected (
                    inputs = self.h1,
                    num_outputs = 1,
                    activation_fn = None,
                    weights_initializer = normc_initializer(1.0),
                    biases_initializer = tf.constant_initializer(0.0)
                ))
            self.theta_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                    inputs = self.h1,
                    num_outputs = 1,
                    activation_fn = None, # tanh for -1 to 1
                    weights_initializer = normc_initializer(1.0),
                    biases_initializer = tf.constant_initializer(-.1*self.n_hidden)
                ))
            self.r_logstd = tf.Variable(-1.)
            self.theta_logstd = tf.Variable(0.)
            self.r_distr = tf.contrib.distributions.Normal(self.r_mean, tf.exp(self.r_logstd))
            self.theta_distr = tf.contrib.distributions.Normal(self.theta_mean, tf.exp(self.theta_logstd))
            self.sy_r_sample = self.r_distr.sample()
            self.sy_theta_sample = self.theta_distr.sample()
            self.x_mean = self.r_mean * tf.cos(self.theta_mean)
            self.y_mean = self.r_mean * tf.sin(self.theta_mean)
            self.sy_x_sample = self.sy_r_sample * tf.cos(self.sy_theta_sample)
            self.sy_y_sample = self.sy_r_sample * tf.sin(self.sy_theta_sample)
        else:
            self.x_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                    inputs = self.h1,
                    num_outputs = 1,
                    activation_fn = None,
                    weights_initializer = normc_initializer(.2),
                    biases_initializer = tf.constant_initializer(0.0)
                ))
            self.y_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                    inputs = self.h1,
                    num_outputs = 1,
                    activation_fn = None,
                    weights_initializer = normc_initializer(.2),
                    biases_initializer = tf.constant_initializer(0.0)
                ))
            self.x_logstd = tf.Variable(-1.)
            self.y_logstd = tf.Variable(-1.)
            self.x_distr = tf.contrib.distributions.Normal(self.x_mean, tf.exp(self.x_logstd))
            self.y_distr = tf.contrib.distributions.Normal(self.y_mean, tf.exp(self.y_logstd))
            self.sy_x_sample = self.x_distr.sample()
            self.sy_y_sample = self.y_distr.sample()

        if self.polar:
            self.r_logprob = self.r_distr.log_prob(tf.sqrt(self.sy_x_actions**2 + self.sy_y_actions**2))
            self.theta_logprob = self.theta_distr.log_prob(tf.atan(self.sy_y_actions / self.sy_x_actions))
            self.sy_surr = - tf.reduce_mean(self.sy_adv * (self.theta_logprob + self.r_logprob))
        else:
            self.x_logprob = self.x_distr.log_prob(self.sy_x_actions)
            self.y_logprob = self.y_distr.log_prob(self.sy_y_actions)
            self.sy_surr = - tf.reduce_mean(self.sy_adv * (self.x_logprob + self.y_logprob))

        self.update_op = tf.train.AdamOptimizer(self.sy_stepsize).minimize(self.sy_surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def reset_accum(self):
        self.xs_accum = np.empty((0, self.n_bits))
        self.x_accum = np.empty(0)
        self.y_accum = np.empty(0)
        self.adv_accum = np.empty(0)

    def policy_update(self):
        print ("updating policy")
        if self.polar:
            theta_logstd, r_logstd = self.sess.run([self.theta_logstd, self.r_logstd])
            print ("theta_std:", np.exp(theta_logstd))
            print ("r_std:", np.exp(r_logstd))
        else:
            x_logstd, y_logstd = self.sess.run([self.x_logstd, self.y_logstd])
            print ("x_std:", np.exp(x_logstd))
            print ("y_std:", np.exp(y_logstd))

        self.sess.run([self.update_op], feed_dict={
                self.sy_x: self.xs_accum,
                self.sy_x_actions: self.x_accum,
                self.sy_y_actions: self.y_accum,
                self.sy_adv: self.adv_accum,
                self.sy_stepsize: self.stepsize
            })

        self.reset_accum()

    def transmit(self, x_input, evaluate=False):
        self.step += 1

        # convert input into proper format (e.g. x=[1 0] --> [1 -1])
        x_input = 2 * (x_input - .5)

        # run policy
        if evaluate:
            x, y = self.sess.run([self.x_mean, self.y_mean], feed_dict={
                    self.sy_x: x_input
            })
        else:
            x, y = self.sess.run([self.sy_x_sample, self.sy_y_sample], feed_dict={
                    self.sy_x: x_input
                })
            self.xs_accum = np.r_[self.xs_accum, x_input]
            self.x_accum = np.r_[self.x_accum, np.array(x)[None]]
            self.y_accum = np.r_[self.y_accum, np.array(y)[None]]

        return np.array([x, y])

    def receive_reward(self, rew):
        self.adv_accum = np.r_[self.adv_accum, rew + 2.]

        # If episode over, update policy and reset
        if self.step >= self.steps_per_episode:
            self.policy_update()
            self.step = 0

    def constellation(self, iteration=0, groundtruth=None):
        """
        Plots a constellation diagram. (https://en.wikipedia.org/wiki/Constellation_diagram)
        """
        bitstrings = list(itertools.product([0, 1], repeat=self.n_bits))

        plt.figure(figsize=(16, 16))
        size = 20

        for bs in bitstrings:
            x,y = self.transmit(np.array(bs)[None], evaluate=True)
            plt.scatter(x, y, label=str(bs))
            plt.annotate(str(bs), (x, y), size=size)
        plt.axvline(0)
        plt.axhline(0)
        plt.xlim([-2., 2.])
        plt.ylim([-2., 2.])

        if groundtruth:
            for k in groundtruth.keys():
                x_gt, y_gt = groundtruth[k]
                plt.scatter(x_gt, y_gt, s=size, color='purple')
                plt.annotate(''.join([str(b) for b in k]), (x_gt, y_gt), size=size)

        plt.savefig('figures/%d.png' % iteration)
        plt.close()

# Given a decoding map, return the closest bitstring
def rx_decode(rx_inp, decoding_map):
    rx_out, dist = None, float("inf")
    for k in decoding_map.keys():
        d = np.linalg.norm(rx_inp - decoding_map[k], ord=2)
        if d < dist:
            rx_out = np.array(k)
            dist = d
    return rx_out

if __name__ == '__main__':
    # set random seeds
    # tf.set_random_seed(0)
    # np.random.seed(0)

    # page 570 of (Proakis, Salehi)
    psk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1, 1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1, 1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([1, -1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([-1,-1])
    }

    qam16 = {
        (0, 0, 0, 0): 0.5/np.sqrt(2)*np.array([1, 1]),
        (0, 0, 0, 1): 0.5/np.sqrt(2)*np.array([3, 1]),
        (0, 0, 1, 0): 0.5/np.sqrt(2)*np.array([1, 3]),
        (0, 0, 1, 1): 0.5/np.sqrt(2)*np.array([3, 3]),
        (0, 1, 0, 0): 0.5/np.sqrt(2)*np.array([1, -1]),
        (0, 1, 0, 1): 0.5/np.sqrt(2)*np.array([1, -3]),
        (0, 1, 1, 0): 0.5/np.sqrt(2)*np.array([3, -1]),
        (0, 1, 1, 1): 0.5/np.sqrt(2)*np.array([3, -3]),
        (1, 0, 0, 0): 0.5/np.sqrt(2)*np.array([-1, 1]),
        (1, 0, 0, 1): 0.5/np.sqrt(2)*np.array([-1, 3]),
        (1, 0, 1, 0): 0.5/np.sqrt(2)*np.array([-3, 1]),
        (1, 0, 1, 1): 0.5/np.sqrt(2)*np.array([-3, 3]),
        (1, 1, 0, 0): 0.5/np.sqrt(2)*np.array([-1, -1]),
        (1, 1, 0, 1): 0.5/np.sqrt(2)*np.array([-3, -1]),
        (1, 1, 1, 0): 0.5/np.sqrt(2)*np.array([-1, -3]),
        (1, 1, 1, 1): 0.5/np.sqrt(2)*np.array([-3, -3])
    }

    # parameters
    decoding_map = psk
    n_bits = int(np.log2(len(decoding_map.keys())))
    l = .1
    steps_per_episode = 4096

    # instantiate environment and transmitter
    env = Environment(n_bits=n_bits, l=l)
    nt = NeuralTransmitter(n_bits=n_bits, steps_per_episode=steps_per_episode, polar=False, stepsize=5e-3)

    # training and evaluation
    for i in range(1000):
        rew_per_ep = 0.0
        start = time.time()
        for _ in range(steps_per_episode):
            # tx
            tx_inp = env.get_input_transmitter()
            tx_out = nt.transmit(tx_inp[None])
            env.output_transmitter(tx_out)

            # rx
            rx_inp = env.get_input_receiver()
            rx_out = rx_decode(rx_inp, decoding_map)
            env.output_receiver(rx_out)

            # rewards
            tx_reward = env.reward_transmitter()
            rx_reward = env.reward_receiver()
            nt.receive_reward(tx_reward)
            rew_per_ep += tx_reward * 1.0/steps_per_episode

        end = time.time()

        # Evaluate on all bitstrings
        bitstrings = list(itertools.product([0, 1], repeat=n_bits))
        rew = 0.0
        for b in bitstrings:
            tx_out = nt.transmit(np.array(b)[None], evaluate=True)
            rx_inp = tx_out
            rx_out = rx_decode(rx_inp, decoding_map)
            rew += np.linalg.norm(np.array(b) - rx_out, ord=1)

        print ("\n######## Epoch %d ########" % i)
        print ("rew_per_ep:", rew_per_ep)
        print ("bits incorrect / %d:" % (n_bits*2**(n_bits)), rew)
        print ("wall clock time: %.4f ms" % ((end - start)*1000))

        if i % 10 == 0:
            nt.constellation(iteration=i, groundtruth=decoding_map)