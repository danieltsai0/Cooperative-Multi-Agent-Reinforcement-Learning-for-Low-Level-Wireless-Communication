import numpy as np
import tensorflow as tf

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
#  Misc utility functions and variables
#
############################################################ 

######################
# Functions for QPSK #
######################

psk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1,-1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1,-1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([-1,1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([1,1])
    }

def qpsk_int_to_coord(x):
    return np.array([x//2, x%2])

def qpsk_coord_to_int(x1, x2):
    return x1 + 2*x2