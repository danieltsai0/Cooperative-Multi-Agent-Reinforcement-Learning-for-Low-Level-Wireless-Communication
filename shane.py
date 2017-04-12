############################################################
#
#  Basic Learning Receiver
#  Shane Barratt
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

# normalized constant initializer from cs 294-112 code
def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class NeuralTransmitter(object):
    def __init__(self, n_bits=4, n_hidden=4, steps_per_episode=32, stepsize=1e-2):
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.steps_per_episode = steps_per_episode
        self.stepsize = stepsize

        self.step = 0 # current step

        # saved xs, actions and advantages
        self.reset_accum()

        # Network Structure
        self.sy_x = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.sy_r = tf.placeholder(tf.float32, [None]) # radius in polar coordinates
        self.sy_theta = tf.placeholder(tf.float32, [None]) # angle in radians
        self.sy_adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation

        # Hidden Layer
        self.h1 = tf.contrib.layers.fully_connected(
            inputs = self.sy_x,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )

        # Outputs
        self.theta = np.pi * tf.contrib.layers.fully_connected(
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = tf.nn.tanh, # tanh for -1 to 1
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            )
        self.r_scale = 2*tf.Variable(.707, name='r_scale')
        self.r = self.r_scale * tf.contrib.layers.fully_connected (
                inputs = self.h1,
                num_outputs = 1,
                activation_fn = tf.nn.sigmoid, # sigmoid for 0 to 1
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            )
        self.r_logstd = tf.Variable(-2.)
        self.theta_logstd = tf.Variable(0.7)

        # randomized actions
        self.r_distr = tf.contrib.distributions.Normal(self.r, tf.exp(self.r_logstd))
        self.theta_distr = tf.contrib.distributions.Normal(self.theta, tf.exp(self.theta_logstd))

        self.sy_r_sample = self.r_distr.sample()
        self.sy_theta_sample = self.theta_distr.sample()

        # for forming surrogate loss
        self.r_logprob = self.r_distr.log_prob(self.sy_r)
        self.theta_logprob = self.theta_distr.log_prob(self.sy_theta)

        self.sy_surr = - tf.reduce_mean(self.sy_adv * self.theta_logprob * self.r_logprob)
        self.sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)
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
        r_scale, theta_logstd, r_logstd = self.sess.run([self.r_scale, self.theta_logstd, self.r_logstd])

        sy_surr_bef = self.sess.run([self.sy_surr], feed_dict={
                self.sy_x: self.xs_accum,
                self.sy_r: self.r_accum,
                self.sy_theta: self.theta_accum,
                self.sy_adv: self.adv_accum,
                self.sy_stepsize: self.stepsize
            })

        print ("sy_surr_bef:", sy_surr_bef)
        print ("r_scale:", r_scale)
        print ("theta_std:", np.exp(theta_logstd))
        print ("r_std:", np.exp(r_logstd))

        self.sess.run([self.update_op], feed_dict={
                self.sy_x: self.xs_accum,
                self.sy_r: self.r_accum,
                self.sy_theta: self.theta_accum,
                self.sy_adv: self.adv_accum,
                self.sy_stepsize: self.stepsize
            })

        sy_surr_aft = self.sess.run([self.sy_surr], feed_dict={
                self.sy_x: self.xs_accum,
                self.sy_r: self.r_accum,
                self.sy_theta: self.theta_accum,
                self.sy_adv: self.adv_accum,
                self.sy_stepsize: self.stepsize
            })

        print ("sy_surr_aft:", sy_surr_aft)

        self.reset_accum()

    def transmit(self, x):
        self.step += 1

        # convert input into proper format (e.g. x=[1 0] --> [1 -1])
        x = 2 * (x - .5)

        # run policy
        theta, r = self.sess.run([self.sy_theta_sample, self.sy_r_sample], feed_dict={
                self.sy_x: x
            })

        self.xs_accum = np.r_[self.xs_accum, x]
        self.r_accum = np.r_[self.r_accum, r[0]]
        self.theta_accum = np.r_[self.theta_accum, theta[0]]

        theta, r = theta[0][0], r[0][0]
        return r * np.array([np.cos(theta), np.sin(theta)])

    def receive_reward(self, rew):
        self.adv_accum = np.r_[self.adv_accum, rew + .1]

        # If episode over, update policy and reset
        if self.step >= self.steps_per_episode:
            self.policy_update()
            self.step = 0

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
    env = Environment(n_bits=2, l=.001)

    psk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1,-1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1,-1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([-1,1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([1,1])
    }

    steps_per_episode = 1000

    nt = NeuralTransmitter(n_bits=2, steps_per_episode=steps_per_episode)

    for i in range(1000):
        rew_per_ep = 0.0
        for _ in range(steps_per_episode):
            # tx
            tx_inp = env.get_input_transmitter()
            tx_out = nt.transmit(tx_inp[None])
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

            nt.receive_reward(tx_reward)

            rew_per_ep += tx_reward * 1.0/steps_per_episode
        print ("rew_per_ep:", rew_per_ep)
    nt.constellation()
