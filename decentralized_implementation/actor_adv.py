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

    """
    Transmit chunk of preamble.

    Returns:
        p_m: modulated preamble - each row of preamble is converted into a single complex number
    """
    def transmit_preamble(self):
        preamble_bit = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        # print(str(self.id_num)+'s index is curretly: '+str(self.index))
        return self.t_unit.transmit(preamble_bit)

    """
    Transmit preamble guess (receiver demodulated signal)
    
    Inputs:
        preamble_g_bit: bit format preamble guess - demodulated by receiver

    Returns:
        p_m: modulated preamble - each row of preamble converted to single complex number
        p_m_g: modulated preamble guess - each row of guessed preamble coverted to single complex number
    """
    def transmit_preamble_g(self, preamble_g_bit):
        preamble_bit = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        return self.t_unit.transmit(preamble_bit), self.t_unit.transmit(preamble_g_bit)


    """
    Updates transmitter with the received preamble guess.

    Inputs:
        preamble_g_g_bit: bit format guess of preamble guess - 
    """
    def transmitter_update(self, preamble_g_g_bit):
        self.t_unit.update(preamble_g_g_bit)

    """
    Gives modulated preamble and preamble labels to the receiver so that receiver can use it 
    to guess the lables for the modulated preamble with Knn or to demodulate the preamble guess
    signal.

    Inputs:
        preamble_mod: modulated premable signal - taken from other Actor's transmitter
    """
    def receive(self, preamble_mod):
        preamble_bit = self.preamble[self.index*self.size_of_episode:(self.index+1)*self.size_of_episode]
        # print(str(self.id_num)+'s index is curretly: '+str(self.index))
        self.r_unit.receive(preamble_mod, preamble_bit)

    """
    Calls receive with the modulated preamble signal and then asks the receiver to guess the 
    labels for each complex input via Knn.

    Inputs:
        preamble_mod: modulated premable signal - taken from other Actor's transmitter

    Returns:
        premable_g_bit: bit format preamble guess
    """
    def receive_preamble(self, preamble_mod):
        self.receive(preamble_mod)
        return self.r_unit.generate_guess()

    """
    Calls receive with the modulated preamble signal and then asks the receiver to guess the 
    labels for the preamble guess signal.

    Inputs:
        preamble_mod: modulated premable signal - taken from other Actor's transmitter
        preamble_g_mod: modulated preamble guess signal - taken from other Actors' transmitter
                        and represents the modulation of the other Actors' receiver's guesses
                        for the first transmission.
    """
    def receive_preamble_g(self, preamble_mod, preamble_g_mod):
        self.receive(preamble_mod)
        return self.r_unit.generate_demod(preamble_g_mod)


    """
    Gets transmitter to visualize the means of its transmissions.

    Inputs:
        iteration: just for bookkeeping when producing images
    """
    def visualize(self, iteration):
        self.t_unit.visualize(iteration)

    """
    Called externally after entire iteration is done.
    """
    def increment_pi(self):
        self.index = (self.index + 1) % self.max_index
