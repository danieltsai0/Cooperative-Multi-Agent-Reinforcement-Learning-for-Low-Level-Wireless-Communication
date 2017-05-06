import transmitter
import receiver
import pickle
import util

class Actor():
    def __init__(self, t_args, r_args, stepsize, dirname):
        self.transmitter = transmitter.NeuralTransmitter(*(t_args+[dirname]))
        self.receiver = receiver.KnnReceiver(*r_args)
        self.init_stepsize = stepsize
        self.dirname = dirname
        util.create_dir(self.dirname)


    def transmit(self, signal_b):
        signal_m = self.transmitter.transmit(signal_b)
        return signal_m

    def transmitter_update(self, signal_b_g_g, i):
        stepsize = self.init_stepsize

        return self.transmitter.policy_update(signal_b_g_g, stepsize)

    def visualize(self, i, p_args):
        self.transmitter.visualize(i, p_args)

    def receive(self, *args):
        signal_b = self.receiver.receive(*args)
        return signal_b

    def save_stats(self):
        centroid_dict, avg_power, avg_hamming = self.transmitter.get_stats()
        # Save stats
        with open(self.dirname+'stats.log', 'w') as output_file:
            output_file.write("### Statistics ###\n\n")
            output_file.write("Average power: "+str(avg_power)+"\n\n")
            output_file.write("Average hamming distance: "+str(avg_hamming)+"\n\n")
        # Dump centroid dictionary
        with open(self.dirname+'centroid.pickle', 'wb') as output_file:
            pickle.dump(centroid_dict, output_file)

    def save_ber(self, signal_b_g_g):
        self.transmitter.save_ber(signal_b_g_g)