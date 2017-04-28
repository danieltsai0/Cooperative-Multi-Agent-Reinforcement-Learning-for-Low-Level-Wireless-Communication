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
from random import randint, uniform
import pprint
import time
import json
   

class System():

    def __init__(self,
                 run_id,
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
        t_args = [self.preamble, groundtruth, n_bits, n_hidden, 
                  lambda_p, initial_logstd]
        r_args = [self.preamble, k]

        # Receiver Parameters
        self.agent_one = actor.Actor(t_args, r_args, stepsize, output_dir+str(run_id)+'/agent_1/')
        self.agent_two = actor.Actor(t_args, r_args, stepsize, output_dir+str(run_id)+'/agent_2/')

        self.channel = Channel(noise_power)    


    """
    Action sequence that defines one full learning cycle. Starts by transmitting preamble (signal_b_1) from agent one
    to agent two. Agent two demodulates a guess of the preamble (signal_b_g_2) and transmits back the preamble and 
    the generated guess of the preamble (signal_b_2, and signal_b_g_2 resp). Agent one receives these two signals and
    demodulates a guess of the guess of the preamble (signal_b_g_g_1) and updates its transmitter with it.
    """
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
        adv = self.agent_one.transmitter_update(signal_b_g_g_1, i)
        # Visualize transmitter
        if (i % plot_every == 0):
            self.agent_one.visualize(i)
        return adv

    """
    Internal function to swap the agents for echoed learning.
    """
    def swap_agents(self):
        temp = self.agent_one
        self.agent_one = self.agent_two
        self.agent_two = temp

    """
    Run learning simulation.
    """
    def run_sim(self):
        for i in range(self.num_iterations):
            adv1 = self.action_sequence(i)
            self.swap_agents()
            adv2 = self.action_sequence(i)
            self.swap_agents()
            print("iteration %d | avg rewards: %8f %8f" % (i, adv1, adv2))

        self.agent_one.save_stats()
        self.agent_two.save_stats()


""" execute a single run with a set of hyperparameters """
def single_run(params):
   
    directory = output_dir+str(params['run_id'])+'/'
    util.create_dir(directory)
    with open(directory+'params.log', 'w') as output_file:
        output_file.write(str(params['run_id']))
        output_file.write('\n\n')
        for key, value in params.items():
            output_file.write("%s: %s\n" % (str(key), str(value)))
        output_file.write('\n')         

    print("Run ID: %d" % params['run_id'])
    print (params)
    print ('\n')

    sys = System(**params)
    return sys.run_sim()
       
""" generate a run ID based on the unix timestamp """
def gen_id():
    return int(time.time()*1e6)
 
""" sample the hyperparameter space for sweeps """
def hyperparam_sweep(general_params, total):
   
    params = [] 
    for t in range(total):
        run = dict(run_id         = gen_id(),
                   n_hidden       = [randint(20, 80)],
                   stepsize       = uniform(1e-4, 1e-2),
                   lambda_p       = uniform(1e-3, 1e-1),
                   initial_logstd = uniform(-4,1),
                   k              = randint(1,6),
                   num_iterations = 2000,
                   len_preamble   = 2**9,
                   n_bits         = 2,
                   noise_power    = uniform(0,1),
                   **general_params)
        params.append(run)

    return params

if __name__ == '__main__':

    np.random.seed(0)
    output_dir = "output/"
    util.create_dir(output_dir)    

    # read plot_every from commandline
    plot_every = 25 
    if len(sys.argv) == 2:
        plot_every = int(sys.argv[1])

    # General params
    general_params = dict(plot_every = plot_every,
                     ) 

    # Hyperparameters 
    params_single = dict(run_id = gen_id(),
                  n_hidden = [40],
                  stepsize = 5e-3,
                  lambda_p = .1,
                  initial_logstd = 0.,
                  k = 3,
                  num_iterations = 1,
                  len_preamble = 2**9,
                  n_bits = 4,
                  noise_power = 0.1,
                  **general_params)
    
    params_sweep = hyperparam_sweep(general_params, 100)

    ##############   
    # SINGLE RUN
    ##############  
    single_run(params_single) 
    
    ###################   
    # PARAMETER SWEEP 
    ##################  
    p = multiprocessing.Pool()
    #p.map(single_run, params_sweep)
