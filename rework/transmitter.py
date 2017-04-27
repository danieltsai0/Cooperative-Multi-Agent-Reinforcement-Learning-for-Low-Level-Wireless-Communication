import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import time
import util

class NeuralTransmitter():
    def __init__(self, 
                 preamble,
                 groundtruth = util.qpsk,
                 n_bits = 2,
                 n_hidden = [32, 20],
                 stepsize = 1e-2,
                 lambda_p = 0.1,
                 initial_logstd = -2.
                 ):

        # Network variables
        self.preamble = preamble
        self.lambda_p = lambda_p
        self.n_bits = n_bits
        self.groundtruth = groundtruth

        # Create image directories
        self.im_dir = 'figures/'+str(np.random.randint(1,1000))+'_'+str(self.n_bits)+'/'
        util.create_dir(self.im_dir)
        self.im_dir += '%04d.png'
        print("im_dir:",self.im_dir)

        # Placeholders for training
        self.input = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.actions = tf.placeholder(tf.float32, [None, 2])
        self.adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.batch_size = tf.placeholder(tf.int32, [])
        # self.stepsize = tf.placeholder(tf.float32, []) # stepsize
    
        # Network definiton
        layers = [self.input]
        for num in n_hidden:
            h = tf.contrib.layers.fully_connected(
                inputs = layers[-1],
                num_outputs = num,
                activation_fn = tf.nn.relu, # relu activation for hidden layer
                weights_initializer = util.normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
            )
            layers.append(h)

        self.action_means = tf.contrib.layers.fully_connected(
                inputs = layers[-1],
                num_outputs = 2,
                activation_fn = None,
                weights_initializer = util.normc_initializer(1.0),
                biases_initializer = tf.constant_initializer(0.)
        )

        self.action_logstds = initial_logstd*tf.Variable(tf.ones(shape=2))
        self.action_logstds = tf.reshape(self.action_logstds, [-1, 1])
        self.action_logstds = tf.tile(self.action_logstds, [1, self.batch_size])
        self.action_logstds = tf.transpose(self.action_logstds)

        self.action_distr = tf.contrib.distributions.MultivariateNormalDiag(
                self.action_means, 
                tf.exp(self.action_logstds))

        self.action_sample = self.action_distr.sample()

        # Compute log-probabilities for gradient estimation
        self.action_logprob = self.action_distr.log_prob(self.actions)
        self.surr = - tf.reduce_mean(self.adv * self.action_logprob)
        self.optimizer = tf.train.AdamOptimizer(stepsize)
        self.update_op = self.optimizer.minimize(self.surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def policy_update(self, signal_b_g_g):
        adv = -self.lasso_loss(signal_b_g_g)
        print("avg reward:",np.average(adv))

        _ = self.sess.run([self.update_op], feed_dict={
                self.input: self.preamble,
                self.actions: self.preamble_mod,
                self.adv: adv,
                self.batch_size: self.preamble.shape[0]
        })


    def transmit(self, signal_b, save=True):
        signal_m = self.sess.run(self.action_sample, feed_dict={
                self.input: signal_b,
                self.batch_size: signal_b.shape[0]
            })

        if save:
            self.preamble_mod = signal_m
        return signal_m 


    def evaluate(self, signal_b):
        # run policy
        return np.squeeze(self.sess.run(self.action_means, feed_dict={
                self.input: signal_b,
                self.batch_size: signal_b.shape[0]
            }))   


    def visualize(self, iteration):
        start_time = time.time()
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
        size = 1000
        scatter_data = 2*(np.random.randint(0,2,[size,self.n_bits])-.5)
        mod_scatter = self.transmit(scatter_data, save=False)
        ax.scatter(mod_scatter[:,0], mod_scatter[:,1], alpha=0.1, color="red")

        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.savefig(self.im_dir % iteration)
        plt.close()


    def lasso_loss(self, signal_b_g_g):
        return np.linalg.norm(self.preamble - signal_b_g_g, ord=1, axis=1) + \
                    self.lambda_p*np.sum(self.preamble_mod**2,axis=1)