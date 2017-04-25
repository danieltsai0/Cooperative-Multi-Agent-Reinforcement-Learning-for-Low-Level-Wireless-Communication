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
    def __init__(self, n_bits, groundtruth):
        # General args
        preamble = generate_preamble(2**12,n_bits) 
        size_of_episode = 2**7
        # Transmitter args
        n_hidden = 20
        stepsize = 1e-2
        l = .1
        # Receiver args
        k = 3

        self.channel_func = lambda x: x + np.random.normal(loc=0.0, scale=.2, size=[size_of_episode,2])
        self.t_args = [n_bits, n_hidden, stepsize, l, groundtruth]
        self.r_args = [n_bits, k]

        self.actor_one = ActorAdvanced(preamble, size_of_episode, self.t_args, self.r_args)
        self.actor_two = ActorAdvanced(preamble, size_of_episode, self.t_args, self.r_args)
        self.run_func = self.run_adv
        print("Running advanced simulation.\n\n\n")



        print("Parameters for Transmitter:",self.t_args)
        print("\n\nParameters for Receiver:",self.r_args)
        print("\n\n")

        # Create channel
        self.channel = Channel(self.channel_func)


    def swap_actors(self):
        temp = self.actor_one
        self.actor_one = self.actor_two
        self.actor_two = temp


    def advanced_trans_sequence(self,i):
        # _m indicates that it is modulated (complex numbers)
        # _b indicates that it is in bit form
        # p_ indicates preamble
        # p_g indicates preamble guess
        # p_g_g indicates guess of the preamble guess
        p_m_one = self.actor_one.transmit_preamble()
        p_g_b_two = self.actor_two.receive_preamble(self.channel.add_noise(p_m_one))
        p_m_two, p_g_m_two = self.actor_two.transmit_preamble_g(p_g_b_two)
        p_g_g_b_one = self.actor_one.receive_preamble_g(self.channel.add_noise(p_m_two), self.channel.add_noise(p_g_m_two))
        self.actor_one.transmitter_update(p_g_g_b_one)
        self.actor_one.visualize(i)

    """
    Increment the index of the actors' preamble index
    """
    def increment_actors_pi(self):
        self.actor_one.increment_pi()
        self.actor_two.increment_pi()

    """
    Run simulation with advanced actors
    """
    def run_adv(self, num_iterations):
        for i in range(num_iterations):
            # Train first transmitter
            self.advanced_trans_sequence(i)
            self.swap_actors()
            print("<><><><><><><><><> swapping actors <><><><><><><><><>")
            # Train second transmitter
            self.advanced_trans_sequence(i)
            self.swap_actors()
            self.increment_actors_pi()
            print("\n><<><><>><><><><><<>\n")
            print("done with iteration:",i)
            print("\n><<><><>><><><><><<>\n")



if __name__ == '__main__':
    np.random.seed(0)
    # Params
    n_bits = 2
    groundtruth = psk
    num_iterations = 2**7

    # Run
    sys = System(n_bits, groundtruth)
    sys.run_func(num_iterations)