import transmitter
import receiver

class Actor():
    def __init__(self, t_args, r_args):
        self.transmitter = transmitter.NeuralTransmitter(*t_args)
        self.receiver = receiver.KnnReceiver(*r_args)


    def transmit(self, signal_b):
        signal_m = self.transmitter.transmit(signal_b)
        return signal_m

    def transmitter_update(self, signal_b_g_g):
        return self.transmitter.policy_update(signal_b_g_g)

    def visualize(self, i):
        self.transmitter.visualize(i)

    def receive(self, *args):
        signal_b = self.receiver.receive(*args)
        return signal_b

