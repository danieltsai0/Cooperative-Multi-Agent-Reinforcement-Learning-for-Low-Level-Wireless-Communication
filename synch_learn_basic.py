############################################################
#
#  Basic Learning Rx and Tx
#  Daniel Tsai <daniel_tsai@berkeley.edu>
#
#  Simulates a learning receiver with a fixed transmitter.
#
#  Assumptions:
#	- loss is GLOBAL
#   - 'k' is known (i.e. the number of bits that the tramsitter
#       is trying to encode in every symbol)
#
############################################################ 

from environment import Environment
from base_receiver import *
from base_transmitter import *
from util import *

import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import math