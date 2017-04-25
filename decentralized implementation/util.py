import numpy as np
import tensorflow as tf
import os

############################################################
#
#  Network utility functions
#
############################################################ 

# Initializer funct (from HW4)
def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# (from HW4)
def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

# (from HW4)
def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)


############################################################
#
#  Misc utility maps
#
############################################################ 

# Useful maps
color_map = {
            0: 'deepskyblue',
            1: 'orangered',
            2: 'm',
            3: 'lime',
            4: 'skyblue',
            5: 'lightsalmon',
            6: 'fuchsia',
            7: 'palegreen',
            8: 'lightskyblue',
            9: 'coral',
            10: 'violet',
            11: 'mediumseagreen',
            12: 'steelblue',
            13: 'maroon',
            14: 'hotpink',
            15: 'springgreen'
        }

qam16_legend_map = {
            0: '(0,0,0,0)',
            1: '(0,0,0,1)',
            2: '(0,0,1,0)',
            3: '(0,0,1,1)',
            4: '(0,1,0,0)',
            5: '(0,1,0,1)',
            6: '(0,1,1,0)',
            7: '(0,1,1,1)',
            8: '(1,0,0,0)',
            9: '(1,0,0,1)',
            10: '(1,0,1,0)',
            11: '(1,0,1,1)',
            12: '(1,1,0,0)',
            13: '(1,1,0,1)',
            14: '(1,1,1,0)',
            15: '(1,1,1,1)',
        }
qpsk_legend_map = {
            0: '(0,0)',
            1: '(0,1)',
            2: '(1,0)',
            3: '(1,1)'
        }

inv_qam16_legend_map = {v: k for k, v in qam16_legend_map.items()}
inv_qpsk_legend_map = {v: k for k, v in qpsk_legend_map.items()}

qpsk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1, 1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1, 1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([1, -1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([-1,-1])
    }

qam16 = {
        (0, 0, 0, 0): np.array([-3, 3])/(3.0*np.sqrt(2)),
        (0, 0, 0, 1): np.array([-3, 1])/(3.0*np.sqrt(2)),
        (0, 0, 1, 0): np.array([-3, -3])/(3.0*np.sqrt(2)),
        (0, 0, 1, 1): np.array([-3, -1])/(3.0*np.sqrt(2)),
        (0, 1, 0, 0): np.array([-1, 3])/(3.0*np.sqrt(2)),
        (0, 1, 0, 1): np.array([-1, 1])/(3.0*np.sqrt(2)),
        (0, 1, 1, 0): np.array([-1, -3])/(3.0*np.sqrt(2)),
        (0, 1, 1, 1): np.array([-1, -1])/(3.0*np.sqrt(2)),
        (1, 0, 0, 0): np.array([3, 3])/(3.0*np.sqrt(2)),
        (1, 0, 0, 1): np.array([3, 1])/(3.0*np.sqrt(2)),
        (1, 0, 1, 0): np.array([3, -3])/(3.0*np.sqrt(2)),
        (1, 0, 1, 1): np.array([3, -1])/(3.0*np.sqrt(2)),
        (1, 1, 0, 0): np.array([1, 3])/(3.0*np.sqrt(2)),
        (1, 1, 0, 1): np.array([1, 1])/(3.0*np.sqrt(2)),
        (1, 1, 1, 0): np.array([1, -3])/(3.0*np.sqrt(2)),
        (1, 1, 1, 1): np.array([1, -1])/(3.0*np.sqrt(2))
    }


############################################################
#
#  Misc utility functions
#
############################################################ 

def int_to_coord(x, n_bits):
    bin_rep = [int(y) for y in "{0:b}".format(x)]
    pad = n_bits - len(bin_rep)

    return np.pad(bin_rep, pad_width=(pad,0), mode='constant', constant_values=0)

def zero_to_neg_one(x):
    return 2 * (x - .5)

def polar_to_cartesian(r, theta):
    return r * np.array([np.cos(theta), np.sin(theta)])

def get_mod_vars(n_bits):
    if n_bits == 2:
        return qpsk, qpsk_legend_map, inv_qpsk_legend_map

    if n_bits == 4:
        return qam16, qam16_legend_map, inv_qam16_legend_map

"""
Outputs:
    preamble: random matrix of shape (size,n_bits) of -1 and 1
"""
def generate_preamble(size, n_bits):
    return 2*(np.random.randint(0,2,[size,n_bits])-.5)

def generate_id():
    return int(np.random.rand()*100000%100000)

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
