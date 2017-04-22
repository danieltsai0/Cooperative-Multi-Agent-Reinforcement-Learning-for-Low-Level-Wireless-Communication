############################################################
#
# An instance of ActorBasic represents either a transmitter
# or a receiver for . Ideally, two instances
# of this class will be able to learn to communicate with each
# other over a given channel.
#
############################################################ 

from transmitter import *
from receiver import *

class ActorBasic():
	"""
	Inputs:
		t_or_x (Boolean), represents whether the actor is a transmitter or receiver
	"""
	def __init__(self, t_or_x, args):
		self.unit = NeuralTransmitter(*args) if t_or_x else KnnReceiver(*args)

	def transmit(self):
		return self.unit.transmit()

	def receive(self, signal):
		return self.unit.receive(signal)

	def visualize(self, iteration):
		self.unit.visualize(iteration)