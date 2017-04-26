from scipy import stats
import numpy as np

class k_means():
    """ implements the k-means algorithm for demodulation """

    def __init__(self, k):
        self.k = k
        self.means = None 
        self.initialized = False
    
    def reset(self, k=None):
        self.means = None
        if (k is not None): 
            self.k = k
 
    def initialize(self, data, hard=False):
        """ implements the k++ initialization algorithm 
            INPUT
                data: complex valued np.array
                hard: enforces maximum instead of pmf sampling
            OUTPUT 
                compley valued np.array of length k   
        """
        N = data.shape[0] # number of data points
        range_N = np.arange(N)
        means = [data[np.random.randint(N)]]
        dists = float('inf')*np.ones(N) # holds distance of all points to closest mean

        for k in range(self.k-1):
            
            # refresh list of minimal distances
            mean = means[-1] 
            dists = np.minimum(dists, abs(data - mean))

            if (hard): # choose next mean according to max distance 
                means.append(data[np.argmax(dists)])
    
            else: # samples next mean according to pmf ~ distance
                probs = dists/float(np.sum(dists))
                pmf = stats.rv_discrete(values=(range_N, probs))
                means.append(data[pmf.rvs(1)])
                 
        means = np.array(means)
        self.means = means
        self.initialized = True 
        return means

    def iterate(self, data, num_iterations):
        """ executes Lloyd's algorithm on the provided data with the initialized means for i iterations
            INPUT
                data: complex valued np.array
                num_iterations: number of iterations
            OUTPUT:
                vector in which each elements determines the mean assigned to the respective symbol
        """

        if (not self.initialized):
            raise Exception("Not initialized. Run initialize() first") 
       
        N = data.shape[0] 
        means = self.means

        for i in range(num_iterations):
            # assign points to mean
                assign = np.argmin(abs(data[:, None] - means[None, :]), axis=1)                
            # recalculate mean                
                matrix = np.zeros([N, self.k], dtype = np.complex64)
                means = np.array([np.mean(data[assign==k]) for k in range(self.k)])
        self.means = means        
        return assign

    def jump_method(self, data, num_iterations, hard, k_max):
        """ executes the jump method to find a feasible k
            INPUT:
                data: complex valued np.array
                num_iterations: number of iterations to run Lloyd's algorithm
                k_max: maximum k to try
            OUTPUT:
                (jump, { ... k: (distortion[k], means[k], assign[k] ...})
                distortion: scalar distortion value for every k
                means: np.array of means for every k
                assign: vector in which each elements determines the mean assigned to the respective symbol
        """
        N = data.shape[0]
        prev_value = 0
        output = {}
        jump = [0]
        for k in range(1, k_max+1):
            self.k = k
            means = self.initialize(data, hard)
            assign = self.iterate(data, num_iterations)
            
            # calculate distortion
            distortion = np.sum(abs(data-self.means[assign])**2)/float(N)
            output[k] = (distortion, np.copy(self.means), assign)
            jump.append(1.0/distortion - prev_value)
            prev_value = 1.0/distortion 

        return (np.array(jump)/float(max(jump)), output)  
