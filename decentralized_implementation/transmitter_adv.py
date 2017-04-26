############################################################
#
#  Advanced Learning Transmitter
#
#  Network takes in a n_bit long sequence of bits and outputs 
#  a continuous distribution over x and y, which denote the
#  two axes in the complex plane.
#
############################################################ 

from util import *
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time


class NeuralTransmitter(object):
    def __init__(self, 
                 preamble,
                 n_bits=2,
                 action_dim=2,
                 num_hidden_per_layer=[64, 32], 
                 steps_per_episode=32, 
                 stepsize=5e-3, 
                 lambda_power=0.1,
                 desired_kl=None, 
                 initial_logstd=-2.): 

        # Network parameters
        self.action_dim = action_dim
        self.n_hidden_layers = len(num_hidden_per_layer)
        self.num_hidden_per_layer = num_hidden_per_layer
        self.desired_kl = desired_kl
        self.initial_logstd = initial_logstd

        self.steps_per_episode = steps_per_episode
        self.stepsize = stepsize
        self.lambda_power = lambda_power # loss rate for power 

        # Misc. parameters
        self.preamble = preamble
        self.n_bits = n_bits

        # for logging images
        self.im_dir = 'figures/'+str(np.randint(1,10))+'_'+str(n_bits)+'/'
        create_dir(self.im_dir)
        self.im_dir += '%04d.png'

        ############################## 
        # Build Network
        ##############################

        self.sy_x = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.sy_actions = tf.placeholder(tf.float32, [None, self.action_dim]) # x actions for gradient calculation
        self.sy_adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # stepsize for gradient step
        self.sy_batch_size = tf.placeholder(tf.int32, [])

        # Hidden Layers
        self.layers = [self.sy_x]
        for i in range(self.n_hidden_layers):
            h = tf.contrib.layers.fully_connected(
                inputs = self.layers[-1],
                num_outputs = self.num_hidden_per_layer[i],
                activation_fn = tf.nn.relu, # relu activation for hidden layer
                weights_initializer = normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(.1)
            )
            self.layers.append(h)

        self.h_last = self.layers[-1]

        # Outputs
        self.action_means = tf.contrib.layers.fully_connected(
                inputs = self.h_last,
                num_outputs = self.action_dim,
                activation_fn = None,
                weights_initializer = normc_initializer(.2),
                biases_initializer = tf.constant_initializer(0.0)
        )

        self.x_y_logstds = self.initial_logstd*tf.Variable(tf.ones(shape=self.action_dim))
        self.x_y_logstds = tf.reshape(self.x_y_logstds, [-1, 1])
        self.x_y_logstds = tf.tile(self.x_y_logstds, [1, self.sy_batch_size])
        self.x_y_logstds = tf.transpose(self.x_y_logstds)

        self.x_y_distr = tf.contrib.distributions.MultivariateNormalDiag(
                self.action_means, 
                tf.exp(self.x_y_logstds))
        self.action_sample = self.x_y_distr.sample()

        # Compute log-probabilities for gradient estimation
        self.x_y_logprob = self.x_y_distr.log_prob(self.sy_actions)
        self.sy_surr = - tf.reduce_mean(self.sy_adv * self.x_y_logprob)
        self.optimizer = tf.train.AdamOptimizer(self.sy_stepsize)
        self.update_op = self.optimizer.minimize(self.sy_surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    """
    Updates the transmitter based on the input, x and y SAMPLE outputs
    and the advantage (which is just the loss right now).
    """
    def policy_update(self):

        _ = self.sess.run([self.update_op], feed_dict={
                self.sy_x: self.xs_accum,
                self.sy_actions: self.actions_accum,
                self.sy_adv: self.adv,
                self.sy_stepsize: self.stepsize,
                self.sy_batch_size: self.actions_accum.shape[0]
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
        self.adv = - self.ridge_loss(preamble_g_g_bit)
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
        bitstrings = list(itertools.product([-1, 1], repeat=self.n_bits))
        
        fig = plt.figure(figsize=(8, 8))
        plt.title('Constellation Diagram', fontsize=20)
        ax = fig.add_subplot(111)
        ax.set(ylabel='imaginary part', xlabel='real part')

        for bs in bitstrings:
            x,y = self.evaluate(np.array(bs)[None])
            label = (np.array(bs)+1)/2
            ax.scatter(x, y, label=str(label), color='purple', marker="d")
            ax.annotate(str(label), (x, y), size=10)
        ax.axvline(0, color='grey')
        ax.axhline(0, color='grey')
        #ax.grid()
    
        if self.groundtruth:
            for k in self.groundtruth.keys():
                x_gt, y_gt = self.groundtruth[k]
                ax.scatter(x_gt, y_gt, s=5, color='purple')
                # ax.annotate(''.join([str(b) for b in k]), (x_gt, y_gt), size=5)
        
        
        # plot modulated preamble
        mod_preamble = self.transmit(self.preamble)
        ax.scatter(mod_preamble[:,0], mod_preamble[:,1], alpha=0.1, color="red")

        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
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
        return np.linalg.norm(self.trans_input - signal, ord=1, axis=1) + self.l*np.sum(self.trans_output**2,axis=1)

    """
    L2 loss

    Inputs:
        signal: bit format signal to be compared to the original input
    """
    def ridge_loss(self, signal):
        return np.linalg.norm(self.trans_input - signal, axis=1) + self.l*np.sum(self.trans_output**2,axis=1)
