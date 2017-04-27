import transmitter
import receiver

class Actor():
    def __init__(self, t_args, r_args, stepsize):
        self.transmitter = transmitter.NeuralTransmitter(*t_args)
        self.receiver = receiver.KnnReceiver(*r_args)
        self.init_stepsize = stepsize


    def transmit(self, signal_b):
        signal_m = self.transmitter.transmit(signal_b)
        return signal_m

    def transmitter_update(self, signal_b_g_g, i):
        stepsize = self.init_stepsize

        return self.transmitter.policy_update(signal_b_g_g, stepsize)

    def visualize(self, i):
        self.transmitter.visualize(i)

    def receive(self, *args):
        signal_b = self.receiver.receive(*args)
        return signal_b

