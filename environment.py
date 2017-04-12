import tensorflow as tf
import numpy as np

class Environment(object):

    def __init__(self, n_bits = 4, l = .01,
                    noise=lambda x: x + np.random.normal(loc=0.0, scale=.1, size=2)):
        self.n_bits = n_bits
        self.state = 0
        self.l = l
        self.noise = noise

    def get_input_transmitter(self):
        self.input = np.random.randint(0, 2, self.n_bits)
        return self.input

    def output_transmitter(self, output):
        if self.state != 0:
            raise Exception('Wrong order')

        assert output.shape[0] == 2, "Wrong length"

        self.tx_output = output
        self.state = 1

    def get_input_receiver(self):
        if self.state != 1:
            raise Exception('Wrong order')

        self.state = 2
        return self.noise(self.tx_output)

    def output_receiver(self, output):
        if self.state != 2:
            raise Exception('Wrong order')

        assert output.shape[0] == 2, "Wrong length"

        self.rx_output = output
        self.state = 3

    def reward_transmitter(self):
        if self.state != 3:
            raise Exception('Wrong order')

        self.state = 4
        return -self.loss()

    def reward_receiver(self):
        if self.state != 4:
            raise Exception('Wrong order')

        self.state = 0
        return -self.loss()

    def loss(self):
        return np.linalg.norm(self.input - self.rx_output, ord=1) + self.l*np.sum(self.tx_output**2)

if __name__ == '__main__':
    env = Environment(n_bits=2)

    psk = {
        (0, 0): 1.0/np.sqrt(2)*np.array([1,-1]),
        (0, 1): 1.0/np.sqrt(2)*np.array([-1,-1]),
        (1, 0): 1.0/np.sqrt(2)*np.array([-1,1]),
        (1, 1): 1.0/np.sqrt(2)*np.array([1,1])
    }

    # tx
    tx_inp = env.get_input_transmitter()
    tx_out = psk.get(tuple(tx_inp))
    env.output_transmitter(tx_out)

    print ("tx_input:", tx_inp)
    print ("tx_out:", tx_out)

    # rx
    rx_inp = env.get_input_receiver() 
    rx_out, dist = None, float("inf")
    for k in psk.keys():
        d = np.linalg.norm(rx_inp - psk[k], ord=2)
        if d < dist:
            rx_out = np.array(k)
            dist = d
    env.output_receiver(rx_out)

    print ("rx_input:", rx_inp)
    print ("rx_out:", rx_out)

    # rewards
    tx_reward = env.reward_transmitter()
    rx_reward = env.reward_receiver()

    print ("reward:", tx_reward)