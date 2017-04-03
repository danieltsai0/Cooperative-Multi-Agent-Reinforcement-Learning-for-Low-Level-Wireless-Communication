from environment import Environment
import tensorflow as tf
import numpy as np
import IPython as ipy
import itertools
import matplotlib.pyplot as plt

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class NeuralTransmitter(object):
    def __init__(self, n_bits=4, n_hidden=4):
        self.n_bits = n_bits
        self.n_hidden = n_hidden

        # Network Structure
        # ith entry=0 --> x_[2*i] = 1
        # ith entry=1 --> x_[2*i+1] = 1
        # e.g. x = [0 1] -> [1 0 0 1]
        self.x_ = tf.placeholder(tf.float32, [None, self.n_bits*2])

        # Hidden Layer
        self.h1 = tf.contrib.layers.fully_connected(
            inputs = self.x_,
            num_outputs = self.n_hidden,
            activation_fn = tf.nn.relu, # relu activation for hidden layer
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(.1)
        )

        # Outputs: polar form mean, polar form logstd
        self.theta_unscaled = tf.contrib.layers.fully_connected(
            inputs = self.h1,
            num_outputs = 1,
            activation_fn = tf.nn.tanh, # tanh for -1 to 1
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(-.1*self.n_hidden)
        )
        self.r_unscaled = tf.contrib.layers.fully_connected(
            inputs = self.h1,
            num_outputs = 1,
            activation_fn = tf.nn.sigmoid, # sigmoid for 0 to 1
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(-.1*self.n_hidden)
        )
        self.theta = self.theta_unscaled * np.pi
        self.r = self.r_unscaled * tf.Variable(0.5, name='r_scale')
        self.theta_logstd = tf.Variable(-3.)
        self.r_logstd = tf.Variable(-8.)

        self.theta_distr = tf.contrib.distributions.Normal(self.theta, tf.exp(self.theta_logstd))
        self.r_distr = tf.contrib.distributions.Normal(self.r, tf.exp(self.r_logstd))

        self.theta_sample = self.theta_distr.sample()
        self.r_sample = self.r_distr.sample()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def transmit(self, x):
        # convert input into proper format (e.g. x=[1 0] --> [0 1 1 0])
        x_converted = np.empty((0, self.n_bits*2))
        for i in range(x.shape[0]):
            x_new = np.zeros(self.n_bits*2)
            for j in range(self.n_bits):
                x_new[2*j] = 0. if x[i, j] else 1.
                x_new[2*j+1] = 1. if x[i, j] else 0.
            x_converted = np.r_[x_converted, x_new[None]]

        # run network
        theta, r = self.sess.run([self.theta_sample, self.r_sample], feed_dict={
                self.x_: x_converted
            })

        return theta, r

    def constellation(self):
        """
        Plots a constellation diagram. (https://en.wikipedia.org/wiki/Constellation_diagram)
        """
        bitstrings = np.array(20*list(itertools.product([0, 1], repeat=self.n_bits)))
        thetas, rs = self.transmit(bitstrings)

        plt.figure()
        for bits, theta, r in zip(bitstrings, thetas, rs):
            plt.scatter(r*np.cos(theta), r*np.sin(theta), label=str(bits))

        plt.legend()
        plt.show()


if __name__ == '__main__':
    nt = NeuralTransmitter(n_bits=2)
    # x = np.array([[0, 1],[1,1]])
    # print (nt.transmit(x))
    nt.constellation()

if 0 and __name__ == '__main__':
    env = Environment(n_bits=2)

    psk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1,-1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1,-1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([-1,1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([1,1])
    }

    # tx
    tx_inp = env.get_input_transmitter()
    tx_out = psk.get(tuple(tx_inp))
    env.output_transmitter(tx_out)

    print ("tx_input:", tx_inp)
    print ("tx_out:", tx_out)

    # rx
    rx_inp = env.get_input_receiver()
    rx_out, dist = None, float("inf")
    for k in psk.keys():
        d = np.linalg.norm(rx_inp - psk[k], ord=2)
        if d < dist:
            rx_out = np.array(k)
            dist = d
    env.output_receiver(rx_out)

    print ("rx_input:", rx_inp)
    print ("rx_out:", rx_out)

    # rewards
    tx_reward = env.reward_transmitter()
    rx_reward = env.reward_receiver()

    print ("reward:", tx_reward)