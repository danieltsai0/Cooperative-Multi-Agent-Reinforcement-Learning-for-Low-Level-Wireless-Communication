import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import time
import util

class NeuralTransmitter():
    def __init__(self, 
                 preamble,
                 restrict_energy = False,
                 groundtruth = util.qpsk,
                 n_bits = 2,
                 n_hidden = [32, 20],
                 lambda_p = 0.1,
                 initial_logstd = -2.,
                 dirname = None
                 ):

        # Network variables
        self.preamble = preamble
        self.restrict_energy = restrict_energy
        self.lambda_p = lambda_p
        self.n_bits = n_bits
        self.groundtruth = groundtruth
        self.dirname = dirname
        self.im_dir = dirname + '%04d.png'
        self.ber_fn = dirname + 'ber.txt'

        # Placeholders for training
        self.input = tf.placeholder(tf.float32, [None, self.n_bits]) # -1 or 1
        self.actions_re = tf.placeholder(tf.float32, [None]) 
        self.actions_im = tf.placeholder(tf.float32, [None])
        self.adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
        self.stepsize = tf.placeholder(shape=[], dtype=tf.float32) 
        self.batch_size = tf.placeholder(tf.int32, [])
    
        # Network definiton
        layers = [self.input]
        for num in n_hidden:
            h = tf.contrib.layers.fully_connected(
                inputs = layers[-1],
                num_outputs = num,
                activation_fn = tf.nn.relu, # relu activation for hidden layer
                weights_initializer = util.normc_initializer(0.2),
                biases_initializer = tf.constant_initializer(0.2)
            )
            layers.append(h)

        self.re_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs = layers[-1],
                num_outputs = 1,
                activation_fn = None,
                weights_initializer = util.normc_initializer(.5),
                biases_initializer = tf.constant_initializer(0.0)
        ))

        self.im_mean = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs = layers[-1],
                num_outputs = 1,
                activation_fn = None,
                weights_initializer = util.normc_initializer(.5),
                biases_initializer = tf.constant_initializer(0.0)
        ))
        
        # if energy is to be restricted normalize outputs 
        if (self.restrict_energy):
            self.max_amplitude = tf.sqrt(tf.reduce_max(self.re_mean**2 + self.im_mean**2))
            self.normalization = tf.nn.relu(self.max_amplitude-1)+1.0  
            self.re_mean /= self.normalization
            self.im_mean /= self.normalization 
        
        self.re_logstd = tf.Variable(initial_logstd)
        self.im_logstd = tf.Variable(initial_logstd)
        self.re_std = tf.exp(self.re_logstd)
        self.im_std = tf.exp(self.im_logstd)
        
        # randomized actions
        self.re_distr = tf.contrib.distributions.Normal(self.re_mean, self.re_std)
        self.im_distr = tf.contrib.distributions.Normal(self.im_mean, self.im_std)

        self.re_sample = self.re_distr.sample()
        self.im_sample = self.im_distr.sample()

        # Compute log-probabilities for gradient estimation
        self.re_logprob = self.re_distr.log_prob(self.actions_re)
        self.im_logprob = self.im_distr.log_prob(self.actions_im)

        self.surr = - tf.reduce_mean(self.adv * (self.re_logprob + self.im_logprob))
        self.optimizer = tf.train.AdamOptimizer(self.stepsize)
        self.update_op = self.optimizer.minimize(self.surr)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def policy_update(self, signal_b_g_g, stepsize):
        adv = - self.lasso_loss(signal_b_g_g)

        _ = self.sess.run([self.update_op], feed_dict={
                self.input: self.input_accum,
                self.actions_re: self.actions_re_accum,
                self.actions_im: self.actions_im_accum,
                self.adv: adv,
                self.stepsize: stepsize,
                self.batch_size: self.input_accum.shape[0]
        })
        
        return np.average(adv)


    def transmit(self, signal_b, save=True):
        re, im  = self.sess.run([self.re_sample, self.im_sample], feed_dict={
                self.input: signal_b,
                self.batch_size: signal_b.shape[0]
            })
   
        if save:
            self.input_accum = signal_b
            self.actions_re_accum = np.squeeze(re)
            self.actions_im_accum = np.squeeze(im)

        signal_m = np.array([np.squeeze(re),np.squeeze(im)]).T
        return signal_m


    def evaluate(self, signal_b):
        # run policy
        re, im = np.squeeze(self.sess.run([self.re_mean, self.im_mean], feed_dict={
                self.input: signal_b,
                self.batch_size: signal_b.shape[0]
            }))   
        return np.array([np.squeeze(re),np.squeeze(im)]).T


    def visualize(self, iteration, p_args=None):
        """
        Plots a constellation diagram. (https://en.wikipedia.org/wiki/Constellation_diagram)
        """
        
        fig = plt.figure(figsize=(8, 8))
        plt.title('Constellation Diagram', fontsize=20)
        ax = fig.add_subplot(111)
        ax.set(ylabel='imaginary part', xlabel='real part')

        # plot modulated preamble
        size = 10000
        scatter_data = 2*(np.random.randint(0,2,[size,self.n_bits])-.5)
        mod_scatter = self.transmit(scatter_data, save=False)
        ax.scatter(mod_scatter[:,0], mod_scatter[:,1], alpha=0.1, color="red")

        # plot means and labels
        bitstrings = list(itertools.product([-1., 1.], repeat=self.n_bits))
        eval_data = self.evaluate(np.array(bitstrings))
        Eb = np.mean(eval_data[:,0]**2 + eval_data[:,1]**2)/float(self.n_bits)        

        for i,bs in enumerate(bitstrings):
            x,y = eval_data[i,0], eval_data[i,1] 
            label = (np.array(bs)+1)/2
            ax.scatter(x, y, label=str(label), color='purple', marker="d")
            ax.annotate(str(label), (x, y), size=10)
            noise_circle = plt.Circle((x,y), np.sqrt(p_args['noise_power']), color='purple', fill=False)
            ax.add_artist(noise_circle)                

        ax.axvline(0, color='grey')
        ax.axhline(0, color='grey')
    
        if self.groundtruth:
            for k in self.groundtruth.keys():
                re_gt, im_gt = self.groundtruth[k]
                ax.scatter(re_gt, im_gt, s=5, color='purple')
                # ax.annotate(''.join([str(b) for b in k]), (re_gt, im_gt), size=5)

        if (self.restrict_energy):
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            # write arguments in graph for easy tuning
            x_text = ax.get_xlim()[0]+0.1
            y_text = ax.get_ylim()[1]-0.75
            unit_circle = plt.Circle((0,0), 1, color='grey', fill=False)
            ax.add_artist(unit_circle)                

        else:   
            plt.xlim([-3, 3])
            plt.ylim([-3, 3])
            # write arguments in graph for easy tuning
            x_text = ax.get_xlim()[0]+0.3
            y_text = ax.get_ylim()[1]-1.4
        
        p_args['$E_b/N_0$'] = 10*np.log10(Eb/p_args['noise_power'])
        param_text = '\n'.join([key+": "+ str(p_args[key]) for key in p_args.keys()])
        ax.text(x_text, y_text, param_text,
                fontsize=9,
                bbox={'facecolor':'white', 'alpha':0.5})

        plt.savefig(self.im_dir % iteration)
        plt.close()


    def save_ber(self, signal_b_g_g):
        ber = np.sum(np.linalg.norm(self.input_accum - signal_b_g_g, ord=1, axis=1)/2)
        with open(self.ber_fn, 'a') as f:
            f.write(str(ber) + "\n")


    def lasso_loss(self, signal_b_g_g):
        if (self.restrict_energy):
            return np.linalg.norm(self.input_accum - signal_b_g_g, ord=1, axis=1)

        # if the energy is unrestricted during training 
        else: 
            return np.linalg.norm(self.input_accum - signal_b_g_g, ord=1, axis=1) + \
                    self.lambda_p*np.average(self.actions_re_accum**2 + self.actions_im_accum**2) + \
                    self.lambda_p*(self.actions_re_accum**2 + self.actions_im_accum**2)

    #####################
    # Methods for stats #
    #####################


    def get_stats(self):
        # Variables
        k = 3
        coords_ary = np.empty((0,2))
        labels_ary = np.empty((0,self.n_bits))
        bitstrings = list(itertools.product([-1, 1], repeat=self.n_bits))
        # Generate centroids
        for bs in bitstrings:
            coords_ary = np.r_[coords_ary,np.array(self.evaluate(np.array(bs)[None]))[None]]
            labels_ary = np.r_[labels_ary,(np.array(bs)[None]+1)/2]

        # Create centroid_dict
        labels_tuple = [tuple(x) for x in labels_ary.tolist()]
        centroid_dict = dict(zip(labels_tuple, coords_ary.tolist()))
        # Calculate avg power
        avg_power = np.average(np.sum(coords_ary**2, axis=1))

        return centroid_dict, avg_power, util.avg_hamming(k, coords_ary, labels_ary)
