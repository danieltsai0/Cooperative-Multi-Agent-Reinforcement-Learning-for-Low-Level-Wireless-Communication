    #### CURRENTLY NOT IMPLEMENTED, JUST CHANGED ARGS FOR ACTORADVANCED TO WORK ####
    # def run_basic(self, num_iterations):
    #   for i in range(num_iterations):
    #       self.transmission_sequence()
    #       print("><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><")
    #       print("done with iteration:",i)
    #       print("><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><><<><><>><><><><")

        # def transmission_sequence(self):
    #     trans = self.actor_one.transmit_re()
    #     self.actor_two.receive(self.channel.add_noise(trans))
    #     trans = self.actor_two.transmit()
    #     self.actor_one.receive(trans)
    #     self.actor_one.visualize(i)

        #### CURRENTLY NOT IMPLEMENTED, JUST CHANGED ARGS FOR ACTORADVANCED TO WORK ####
        # else:
        #   self.actor_one = ActorBasic(True, self.t_args) # transmitter
        #   self.actor_two = ActorBasic(False, self.r_args) # receiver
        #   run_func = self.run_basic
        #   print("Running basic simulation.\n\n\n")

    
"""
Inputs:
    a: a numpy array of size (?,num_bits)
    label: the correct label of the transmission

Outputs:
    percentage of labels that were correct in a
"""
# def percent_correct(a, label):
#     a = np.ascontiguousarray(a)
#     void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
#     a = a.view(void_dt).ravel()
#     label = label.view(void_dt).ravel()
#     return np.count_nonzero(a==label) / a.shape[0]