############################################################
#
# Can be used to implement time varying noise or just constant
# random noise.
#
############################################################ 


def white_noise(loc, scale):
	return lambda x: x + np.random.normal(loc, scale, size=2)


class Channel():
	def __init__(self, function):
		self.noise_function = function

	def add_noise(self, transmission):
		return self.noise_function(transmission)
