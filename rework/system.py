import numpy as np
import actor
import channel
import util


class System():

    def __init__(self, 
                    num_iterations, len_preamble,
                    groundtruth, n_bits, n_actions, n_hidden, stepsize, lambda_p, initial_logstd,
                    k,
                    channel_func):


        # System Parameters
        self.num_iterations = num_iterations
        self.preamble = util.generate_preamble(len_preamble, n_bits)

        # Transmitter Parameters
        t_args = [self.preamble, groundtruth, n_bits, n_hidden, stepsize, 
                        lambda_p, initial_logstd]
        r_args = [self.preamble, k]

        # Receiver Parameters

        self.agent_one = actor.Actor(t_args, r_args)
        self.agent_two = actor.Actor(t_args, r_args)

        self.channel_noise = channel.Channel(channel_func).add_noise


    def action_sequence(self, i):
        # Compute signal_b here
        signal_b = self.preamble
        # Transmit bit signal, produce modulated signal
        signal_m_1 = self.agent_one.transmit(signal_b)
        # Apply channel noise, produce noisy modulated signal
        signal_m_1 = self.channel_noise(signal_m_1) 
        # Receive mod signal, produce bit signal guess
        signal_b_g_2 = self.agent_two.receive(signal_m_1)
        # Transmit bit signal guess, produce mod signal and mod signal guess as tuple
        signal_m_2, signal_m_g_2 = self.agent_two.transmit(signal_b), self.agent_two.transmit(signal_b_g_2)
        # Apply channel noise, produce noisy modulated signal
        signal_m_2, signal_m_g_2 = self.channel_noise(signal_m_2), self.channel_noise(signal_m_g_2)
        # Receive mod signal guess, produce bit signal guess of guess
        signal_b_g_g_1 = self.agent_one.receive(signal_m_2, signal_m_g_2)
        # Update transmitter with bit signal guess of guess
        self.agent_one.transmitter_update(signal_b_g_g_1)
        # Visualize transmitter
        if i % 10 == 0:
            self.agent_one.visualize(i)

    def swap_agents(self):
        temp = self.agent_one
        self.agent_one = self.agent_two
        self.agent_two = temp


    def run_sim(self):
        for i in range(self.num_iterations):
            self.action_sequence(i)
            self.swap_agents()
            self.action_sequence(i)
            self.swap_agents()

            print("<><><><><><><><><><>")
            print("done with iteration:",i)
            print("<><><><><><><><><><>")


if __name__ == '__main__':
    np.random.seed(0)

    # System args
    num_iterations = 1000
    len_preamble = 1500
    n_bits = 4
    groundtruth = util.qam16
    n_actions = 1
    n_hidden = [20]
    stepsize = 1e-2
    lambda_p = 0.25
    initial_logstd = -2.
    k = 3
    channel_func = lambda x: x + np.random.normal(loc=0.0, scale=.2, size=[len_preamble,2])


    sys = System(num_iterations, len_preamble,
                 groundtruth, n_bits, n_actions, n_hidden, stepsize, lambda_p, initial_logstd,
                 k,
                 channel_func)

    sys.run_sim()
