
class Channel():
    def __init__(self, function):
        self.noise_function = function

    def add_noise(self, transmission):
        return self.noise_function(transmission)
