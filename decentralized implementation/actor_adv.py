############################################################
#
# An instance of ActorAdvanced contains both a transmitter
# and a receiver for synchronous training. Ideally, two instances
# of this class will be able to learn to communicate with each
# other over a given channel.
#
############################################################ 

from abc import *
from transmitter_adv import *
from receiver_adv import *


class ActorAdvanced():
    """
    Inputs:
        t_args, represents initializer arguments for the transmitter
        r_args, represents initializer arguments for the receiver
    """
    def __init__(self, preamble, size_of_episode, t_args, r_args):
        self.preamble = preamble
        self.id_num = generate_id()
        print("id number:",self.id_num)
        self.t_unit = NeuralTransmitter(*(t_args+[self.id_num]))
        self.r_unit = KnnReceiver(*r_args)

        # values for keeping track of where we are in the preamble
        self.size_of_episode = size_of_episode
        self.index = 0
        self.max_index = self.preamble.shape[0] // size_of_episode

    def transmit_preamble(self):
        preamble_bit = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        print(str(self.id_num)+'s index is curretly: '+str(self.index))
        return self.t_unit.transmit(preamble_bit)

    def transmit_preamble_g(self, preamble_g_bit):
        preamble_bit = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        return self.t_unit.transmit(preamble_bit), self.t_unit.transmit(preamble_g_bit)

    def transmitter_update(self, preamble_g_g_bit):
        self.t_unit.update(preamble_g_g_bit)

    def receive(self, preamble_mod):
        preamble_bit = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        print(str(self.id_num)+'s index is curretly: '+str(self.index))
        self.r_unit.receive(preamble_mod, preamble_bit)

    def receive_preamble(self, preamble_mod):
        self.receive(preamble_mod)
        return self.r_unit.generate_guess()

    def receive_preamble_g(self, preamble_mod, preamble_g_mod):
        self.receive(preamble_mod)
        return self.r_unit.generate_demod(preamble_g_mod)

    def visualize(self, iteration):
        self.t_unit.visualize(iteration)

    """
    Called externally after entire iteration is done.
    """
    def increment_pi(self):
        self.index = (self.index + 1) % self.max_index
