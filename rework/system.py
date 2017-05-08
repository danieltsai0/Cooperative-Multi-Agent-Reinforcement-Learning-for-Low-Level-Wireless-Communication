################################################################################
#
#  Decentralized Multi-Agent Learning Environment
#
#  Can perform a single run or a sweep over parameter.
#  Each run gets a unique ID and its parameters, results
#  and constellation points (pickle + plot) are saved in 
#  a folder named after the ID in the *output* directory.
#
#  Data from runs which score below a certain loss threshold gets moved to 
#  a *discard* directory. All final constellation diagram plots from successfull
#  runs are exported in the *preview* directory for easy inspection.
#
################################################################################

import argparse
import tensorflow as tf
import numpy as np
import actor
from channel import Channel
import util
import multiprocessing
import sys
from random import randint, uniform, sample
import pprint
import time
import json
import shutil   

class System():

    def __init__(self,
                 directory,
                 run_id,
                 plot_every, 
                 restrict_energy,
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
        self.restrict_energy  = restrict_energy
    
        # Transmitter Parameters
        groundtruth = util.schemes[n_bits]
        t_args = [self.preamble, restrict_energy, groundtruth, n_bits, n_hidden, 
                  lambda_p, initial_logstd]
        r_args = [self.preamble, k]

        # Receiver Parameters
        self.agent_one = actor.Actor(t_args, r_args, stepsize, directory+'agent_1/')
        self.agent_two = actor.Actor(t_args, r_args, stepsize, directory+'agent_2/')

        # Parameters to write in the plotted diagrams
        p_args_names = 'run_id total_iters len_preamble stepsize lambda_p initial_logstd noise_power restrict_energy'.split()
        p_args_params = [run_id, num_iterations, len_preamble, stepsize, lambda_p, initial_logstd, noise_power, restrict_energy]
        self.p_args = dict(zip(p_args_names, p_args_params)) 

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
        self.agent_one.save_energy(signal_m_1)
        # Apply channel noise, produce noisy modulated signal
        signal_m_1 = self.channel.AWGN(signal_m_1) 
        # Receive mod signal, produce bit signal guess
        signal_b_g_2 = self.agent_two.receive(signal_m_1)
        # Transmit bit signal guess, 
        # produce mod signal and mod signal guess as tuple
        signal_m_2 = self.agent_two.transmit(signal_b)
        signal_m_g_2 = self.agent_two.transmit(signal_b_g_2)
        # Apply channel noise, produce noisy modulated signal
        signal_m_2, signal_m_g_2 = self.channel.AWGN(signal_m_2), self.channel.AWGN(signal_m_g_2)
        # Receive mod signal guess, produce bit signal guess of guess
        signal_b_g_g_1 = self.agent_one.receive(signal_m_2, signal_m_g_2)
        # Save BER of transmitter one

        self.agent_one.save_ber(signal_b_g_g_1)
        # Update transmitter with bit signal guess of guess
        adv = self.agent_one.transmitter_update(signal_b_g_g_1, i)
        # Visualize transmitter
        if ((i+1) % plot_every == 0):
            self.agent_one.visualize(i+1, self.p_args)
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
    def run_sim(self, verbose):
       
        threshold = -10 # reward threshold to discard runs 
        for i in range(self.num_iterations+1):
            adv1 = self.action_sequence(i)
            self.swap_agents()
            adv2 = self.action_sequence(i)
            self.swap_agents()
            if (verbose):
                print("iteration %d | avg rewards: %8f %8f" % (i, adv1, adv2))

        self.agent_one.save_stats()
        self.agent_two.save_stats()

        # return if the final reward is over below threshold 
        if (adv1 < threshold or adv2 < threshold):
            return False
        
        return True 

""" execute a single run with a set of hyperparameters """
def single_run(params, verbose=False):

    directory = output_dir+"N_"+str(params['noise_power'])+ \
                    "_P_"+str(np.log2(params['len_preamble']))+'/'+ \
                    str(seed)+'/'
    util.create_dir(directory)
    with open(directory+'params.log', 'w') as output_file:
        output_file.write(str(params['run_id']))
        output_file.write('\n\n')
        for key, value in params.items():
            output_file.write("%s: %s\n" % (str(key), str(value)))
        output_file.write('\n')         

    if (verbose):
        print("Run ID: %d" % params['run_id'])
        print (params)
        print ('\n')

    sys = System(directory, **params)
    reward_constraint = sys.run_sim(verbose)

    # if reward constraints not met -> discard
    if (reward_constraint == False):
        shutil.move(directory, discard_dir+str(params['run_id'])+'/')
    else: # copy final plot to preview directory    
        filename = "%04d.png" % (int(params['num_iterations']/plot_every)*plot_every)
        new_filename = str(params['run_id'])
        shutil.copy(directory+'agent_1/'+filename, preview_dir+new_filename+'_1.png')
        shutil.copy(directory+'agent_2/'+filename, preview_dir+new_filename+'_2.png')
  
    with iterations.get_lock():
        iterations.value += 1
    print ("runs completed: %d" % iterations.value) 
     
""" generate a run ID based on the unix timestamp """
def gen_id():
    return int(time.time()*1e6)
 
""" sample the hyperparameter space for sweeps """
def hyperparam_sweep(general_params, total):
   
    params = [] 
    for t in range(total):
        run = dict(run_id         = gen_id(),
                   n_hidden       = [40],
                   stepsize       = uniform(1e-4, 1e-2),
                   lambda_p       = uniform(1e-3, 1e-1),
                   initial_logstd = uniform(-1.5,1),
                   k              = 3,
                   num_iterations = 2000,
                   len_preamble   = 2**randint(7,9),
                   n_bits         = 4,
                   noise_power    = 0.2,
                   **general_params)

        params.append(run)

    return params


def noise_and_preamble_sweep(general_params, noise, preamble_len):
   
    params = [] 
    for n in noise:
        run = dict(run_id = gen_id(),
                  n_hidden = [40],
                  stepsize = 2e-3,
                  lambda_p = 5e-2,
                  initial_logstd = -2.0,
                  k = 3,
                  num_iterations = 2000,
                  len_preamble = 2**9,
                  n_bits = 4,
                  noise_power = n,
                  **general_params)

        params.append(run)

    # for p in preamble_len:
    #     run = dict(run_id = gen_id(),
    #                n_hidden = [40],
    #               stepsize = 2.5e-3,
    #               lambda_p = 5e-2,
    #               initial_logstd = -1.0,
    #               k = 3,
    #               num_iterations = 2000,
    #               len_preamble = p,
    #               n_bits = 4,
    #               noise_power = 0.1,
    #               **general_params)

    #     params.append(run)

    return params

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument("seed", type=int,
             help="seed for randomness")
    args, leftovers = parser.parse_known_args()

    iterations = multiprocessing.Value('i', 0)
    seed = args.seed
    print("running with seed: %d" % seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    output_dir = "output/"
    discard_dir = output_dir+"shitty/" # runs that don't meet the loss threshold
    preview_dir = output_dir+"preview/" # folder for all final constellations
    util.create_dir(output_dir)    
    util.create_dir(discard_dir)
    util.create_dir(preview_dir)

    # read plot_every from commandline
    plot_every = 1000 
    # if len(sys.argv) == 2:
    #     plot_every = int(sys.argv[1])


    # General params
    general_params = dict(plot_every = plot_every,
                          restrict_energy = True 
                     ) 

    # # Hyperparameters for restricted
    # params_single = dict(run_id = gen_id(),
    #               n_hidden = [40],
    #               stepsize = 4e-3,
    #               lambda_p = 0,
    #               initial_logstd = -2.0,
    #               k = 3,
    #               num_iterations = 1000,
    #               len_preamble = 2**8,
    #               n_bits = 4,
    #               noise_power = 0.1,
    #               **general_params)

    # Hyperparameters for unrestricted
    params_single = dict(run_id = gen_id(),
                  n_hidden = [40],
                  stepsize = 2.45e-3,
                  lambda_p = 9e-2,
                  initial_logstd = -1.0,
                  k = 3,
                  num_iterations = 2000,
                  len_preamble = 2**9,
                  n_bits = 4,
                  noise_power = 0.04,
                  **general_params)
    

    noise = [0.01, 0.04, 0.09, 0.16]
    preamble_len = [2**7, 2**8, 2**9]
    params_sweep = noise_and_preamble_sweep (general_params, noise, preamble_len)

    #############
    # SWITCH
    run_sweep = True 

    ##############   
    # SINGLE RUN
    ##############  
    if not run_sweep:
        single_run(params_single, verbose=False) 
    
    ###################   
    # PARAMETER SWEEP 
    ##################  
    else:
        print ("Executing parameter sweep with %d runs" % len(params_sweep))
        p = multiprocessing.Pool()
        p.map(single_run, params_sweep)
