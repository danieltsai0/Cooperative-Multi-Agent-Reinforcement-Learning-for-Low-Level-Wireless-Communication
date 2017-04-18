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
    def __init__(self, n_bits=2, n_hidden=16, steps_per_episode=32, stepsize=1e-3):
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.steps_per_episode = steps_per_episode
        self.stepsize = stepsize

        self.step = 0 # current step

        # saved xs, actions and advantages
        self.reset_accum()

        # Network
        self.sy_x = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.sy_r = tf.placeholder(tf.float32, [None]) # radius in polar coordinates
        self.sy_theta = tf.placeholder(tf.float32, [None]) # angle in radians
        self.sy_adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)

        # Hidden Layer
        self.h1 = tf.contrib.layers.fully_connected(
            inputs = self.sy_x,
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
                biases_initializer = tf.constant_initializer(.7)
            ))
        self.theta_mean = tf.squeeze(np.pi * tf.contrib.layers.fully_connected(
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = tf.nn.tanh, # tanh for -1 to 1
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            ))
        self.r_logstd = tf.Variable(-2.)
        self.theta_logstd = tf.Variable(-.5)
        self.r_std = tf.exp(self.r_logstd)
        self.theta_std = tf.exp(self.theta_logstd)

        # randomized actions
        self.r_distr = tf.contrib.distributions.Normal(self.r_mean, tf.exp(self.r_logstd))
        self.theta_distr = tf.contrib.distributions.Normal(self.theta_mean, tf.exp(self.theta_logstd))

        self.sy_r_sample = self.r_distr.sample()
        self.sy_theta_sample = self.theta_distr.sample()

        # for forming surrogate loss
        self.r_logprob = self.r_distr.log_prob(self.sy_r)
        self.theta_logprob = self.theta_distr.log_prob(self.sy_theta)
        # self.r_logprob = -.5*tf.log(2*np.pi*self.r_std**2) - (self.sy_r - self.r_mean)**2/(2*self.r_std**2)
        # self.theta_logprob = -.5*tf.log(2*np.pi*self.theta_std**2) - (self.sy_theta - self.theta_mean)**2/(2*self.theta_std**2)

        self.sy_surr = - tf.reduce_mean(self.sy_adv * (self.theta_logprob + self.r_logprob))
        self.update_op = tf.train.AdamOptimizer(self.sy_stepsize).minimize(self.sy_surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def reset_accum(self):
        self.xs_accum = np.empty((0, self.n_bits))
        self.r_accum = np.empty(0)
        self.theta_accum = np.empty(0)
        self.adv_accum = np.empty(0)

    def policy_update(self):
        print ("updating policy")
        theta_logstd, r_logstd = self.sess.run([self.theta_logstd, self.r_logstd])

        print ("theta_std:", np.exp(theta_logstd))
        print ("r_std:", np.exp(r_logstd))

        self.sess.run([self.update_op], feed_dict={
                self.sy_x: self.xs_accum,
                self.sy_r: self.r_accum,
                self.sy_theta: self.theta_accum,
                self.sy_adv: self.adv_accum,
                self.sy_stepsize: self.stepsize
            })

        self.reset_accum()

    def transmit(self, x, evaluate=False):
        self.step += 1

        # convert input into proper format (e.g. x=[1 0] --> [1 -1])
        x = 2 * (x - .5)

        # run policy
        if evaluate:
            theta, r = self.sess.run([self.theta_mean, self.r_mean], feed_dict={
                    self.sy_x: x
            })
        else:
            theta, r = self.sess.run([self.sy_theta_sample, self.sy_r_sample], feed_dict={
                    self.sy_x: x
                })

        if not evaluate:
            self.xs_accum = np.r_[self.xs_accum, x]
            self.r_accum = np.r_[self.r_accum, np.array(r)[None]]
            self.theta_accum = np.r_[self.theta_accum, np.array(theta)[None]]

        return r * np.array([np.cos(theta), np.sin(theta)])

    def receive_reward(self, rew):
        self.adv_accum = np.r_[self.adv_accum, rew]

        # If episode over, update policy and reset
        if self.step >= self.steps_per_episode:
            self.policy_update()
            self.step = 0

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

        plt.savefig('figures/%d.png' % iteration)

if __name__ == '__main__':
    # set random seeds
    tf.set_random_seed(0)
    np.random.seed(0)

    n_bits = 4
    l = .01
    steps_per_episode = 512

    env = Environment(n_bits=n_bits, l=l)

    # page 570 of (Proakis, Salehi)
    psk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1, 1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1, 1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([1, -1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([-1,-1])
    }

    qam16 = {
        (0, 0, 0, 0): 1.0/np.sqrt(2)*np.array([1, 1]),
        (0, 0, 0, 1): 1.0/np.sqrt(2)*np.array([2, 1]),
        (0, 0, 1, 0): 1.0/np.sqrt(2)*np.array([1, 2]),
        (0, 0, 1, 1): 1.0/np.sqrt(2)*np.array([2, 2]),
        (0, 1, 0, 0): 1.0/np.sqrt(2)*np.array([1, -1]),
        (0, 1, 0, 1): 1.0/np.sqrt(2)*np.array([1, -2]),
        (0, 1, 1, 0): 1.0/np.sqrt(2)*np.array([2, -1]),
        (0, 1, 1, 1): 1.0/np.sqrt(2)*np.array([2, -2]),
        (1, 0, 0, 0): 1.0/np.sqrt(2)*np.array([-1, 1]),
        (1, 0, 0, 1): 1.0/np.sqrt(2)*np.array([-1, 2]),
        (1, 0, 1, 0): 1.0/np.sqrt(2)*np.array([-2, 1]),
        (1, 0, 1, 1): 1.0/np.sqrt(2)*np.array([-2, 2]),
        (1, 1, 0, 0): 1.0/np.sqrt(2)*np.array([-1, -1]),
        (1, 1, 0, 1): 1.0/np.sqrt(2)*np.array([-2, -1]),
        (1, 1, 1, 0): 1.0/np.sqrt(2)*np.array([-1, -2]),
        (1, 1, 1, 1): 1.0/np.sqrt(2)*np.array([-2, -2])
    }

    def rx_decode(rx_inp, m=psk):
        rx_out, dist = None, float("inf")
        for k in m.keys():
            d = np.linalg.norm(rx_inp - m[k], ord=2)
            if d < dist:
                rx_out = np.array(k)
                dist = d
        return rx_out

    nt = NeuralTransmitter(n_bits=n_bits, steps_per_episode=steps_per_episode)

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
            rx_out = rx_decode(rx_inp, qam16)
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
            rx_out = rx_decode(rx_inp, qam16)
            rew += np.linalg.norm(np.array(b) - rx_out, ord=1)

        print ("\n######## Epoch %d ########" % i)
        print ("rew_per_ep:", rew_per_ep)
        print ("bits incorrect / %d:" % (n_bits*2**(n_bits)), rew)
        print ("wall clock time: %.4f ms" % ((end - start)*1000))

        if i % 10 == 0:
            nt.constellation(iteration=i, groundtruth=qam16)