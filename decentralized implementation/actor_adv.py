############################################################
#
# An instance of ActorAdvanced contains both a transmitter
# and a receiver for synchronous training. Ideally, two instances
# of this class will be able to learn to communicate with each
# other over a given channel.
#
############################################################ 

from abc import *
class ActorAdvanced():
	"""
	Inputs:
		t_or_x (Boolean), represents whether the actor is a transmitter or receiver
	"""
	def __init__(self):
		return