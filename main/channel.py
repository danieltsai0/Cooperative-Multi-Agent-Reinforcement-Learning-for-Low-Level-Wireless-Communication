################################################################################
# 
#  simulates the communication channel 
# 
################################################################################

import numpy as np


class Channel():
    def __init__(self, noise_power):
        self.noise_power = noise_power

    def AWGN(self, signal):
        noise = np.random.normal(loc=0.0, 
                     scale=np.sqrt(self.noise_power/2.0),
                     size=signal.shape)

        return signal + noise
