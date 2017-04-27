################################################################################
#
#  Decentralized Multi-Agent Learning Environment
#
################################################################################

import numpy as np
import actor
from channel import Channel
import util
import multiprocessing
import sys

class System():

    def __init__(self,
                 plot_every, 
                 num_iterations, 
                 len_preamble, 
                 n_bits, 
                 n_hidden, 
                 stepsize, 
                 lambda_p, 
                 initial_logstd,
                 k,
                 noise_power):

        # System Parameters
        self.num_iterations = num_iterations
        self.preamble = util.generate_preamble(len_preamble, n_bits)

        # Transmitter Parameters
        groundtruth = util.schemes[n_bits]
        t_args = [self.preamble, groundtruth, n_bits, n_hidden, stepsize, 
                  lambda_p, initial_logstd]
        r_args = [self.preamble, k]

        # Receiver Parameters
        self.agent_one = actor.Actor(t_args, r_args)
        self.agent_two = actor.Actor(t_args, r_args)

        self.channel = Channel(noise_power)    


    def action_sequence(self, i):
        # Compute signal_b here
        signal_b = self.preamble
        # Transmit bit signal, produce modulated signal
        signal_m_1 = self.agent_one.transmit(signal_b)
        # Apply channel noise, produce noisy modulated signal
        signal_m_1 = self.channel.AWGN(signal_m_1) 
        # Receive mod signal, produce bit signal guess
        signal_b_g_2 = self.agent_two.receive(signal_m_1)
        # Transmit bit signal guess, 
        # produce mod signal and mod signal guess as tuple
        signal_m_2, signal_m_g_2 = self.agent_two.transmit(signal_b), self.agent_two.transmit(signal_b_g_2)
        # Apply channel noise, produce noisy modulated signal
        signal_m_2, signal_m_g_2 = self.channel.AWGN(signal_m_2), self.channel.AWGN(signal_m_g_2)
        # Receive mod signal guess, produce bit signal guess of guess
        signal_b_g_g_1 = self.agent_one.receive(signal_m_2, signal_m_g_2)
        # Update transmitter with bit signal guess of guess
        adv = self.agent_one.transmitter_update(signal_b_g_g_1)

        # Visualize transmitter
        if (i % plot_every == 0):
            self.agent_one.visualize(i)

        return adv

    def swap_agents(self):
        temp = self.agent_one
        self.agent_one = self.agent_two
        self.agent_two = temp

    def run_sim(self):
        for i in range(self.num_iterations):
            adv1 = self.action_sequence(i)
            self.swap_agents()
            adv2 = self.action_sequence(i)
            self.swap_agents()

            print("iteration %d | avg rewards: %8f %8f" % (i, adv1, adv2))


def single_run(params):
    sys = System(**params)
    return sys.run_sim()
        

if __name__ == '__main__':

    np.random.seed(0)

    # read plot_every from commandline
    plot_every = 10
    if len(sys.argv) == 2:
        plot_every = int(sys.argv[1])

    # General params
    general_params = dict(plot_every = plot_every,
                     ) 

    # Hyperparameter 
    params = [dict(n_hidden = [20],
                  stepsize = 1e-2,
                  lambda_p = .1,
                  initial_logstd = 0.,
                  k = 3,
                  num_iterations = 2000,
                  len_preamble = 200,
                  n_bits = 2,
                  noise_power = 0.2,
                  **general_params)
              ] 

    single_run(params[0]) 
     
    #p = multiprocessing.Pool()
    #p.map(single_run, params)


