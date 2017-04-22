from actor_basic import *
from actor_adv import *
from transmitter import *
from receiver import *
from channel import *
from util import *


class System():
	"""
	Inputs:
		a_or_b (Boolean), represents whether we are using advanced or basic Actors
	"""
	def __init__(self, a_or_b, n_bits, groundtruth, num_iterations):
		# General args
		preamble = generate_preamble(2**14,n_bits) 
		size_of_episode = 2**7
		# Transmitter args
		n_hidden = 20
		stepsize = 1e-2
		l = .01
		# Receiver args
		k = 3
		l_or_p = True

		self.channel_func = lambda x: x + np.random.normal(loc=0.0, scale=.2, size=[size_of_episode,2])
		self.actor_one = None
		self.actor_two = None
		if a_or_b:
			self.actor_one = ActorAdvanced()
			self.actor_two = ActorAdvanced()
		else:
			t_args = [preamble, size_of_episode, n_bits, n_hidden, stepsize, l, groundtruth]
			r_args = [preamble, size_of_episode, n_bits, k, l_or_p]
			self.actor_one = ActorBasic(True, t_args) # transmitter
			self.actor_two = ActorBasic(False, r_args) # receiver

		self.channel = Channel(self.channel_func)
		self.run(num_iterations)

	def run(self, num_iterations):
		for i in range(num_iterations):
			trans = self.actor_one.transmit()
			self.actor_two.receive(self.channel.add_noise(trans))
			trans = self.actor_two.transmit()
			self.actor_one.receive(trans)
			self.actor_one.visualize(i)
			print("><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><")
			print("done with iteration:",i)
			print("><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><")


if __name__ == '__main__':
	# Params
	a_or_b = False
	n_bits = 2
	groundtruth = psk
	num_iterations = 100

	# Run
	sys = System(a_or_b, n_bits, groundtruth, num_iterations)