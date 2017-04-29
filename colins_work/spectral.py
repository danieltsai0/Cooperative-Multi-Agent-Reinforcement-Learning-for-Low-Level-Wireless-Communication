from scipy.spatial.distance import pdist, squareform

import scipy as sci
import numpy as np

class Spectral_Analyser():
    """ performs a spectral analysis to find clusters """ 
    #TODO implement kd tree

    def __init__(self):
        pass
    
    def epsilon_neighborhood_graph(self, data, epsilon):
        """ returns an unweighted epsilon neighborhood graph """        

        data_tup = np.stack((data.real, data.imag), axis=1)
        pairwise_dists = squareform(pdist(data_tup, 'euclidean'))
        G = (pairwise_dists<epsilon).astype(int)
        return G

    def knn_graph(self, data, k, mutual):
        """ returns a (mutual) k-nearest neighbour graph """

        data_tup = np.stack((data.real, data.imag), axis=1)
        N = data_tup.shape[0]
        range_N = np.repeat(np.arange(N),k)
        pairwise_dists = squareform(pdist(data_tup, 'euclidean'))
        # get knn indicies per row, column 
        hor_array_sort = np.reshape(np.argsort(pairwise_dists, axis=1)[:,1:k+1], -1, order='C')
        vert_array_sort = np.reshape(np.argsort(pairwise_dists, axis=0)[1:k+1,:], -1, order='F')

        # get data submatricies with only knn in row, column
        hor_knn_data = np.zeros([N,N])
        hor_knn_data[range_N, hor_array_sort] = pairwise_dists[range_N, hor_array_sort]
        vert_knn_data = np.zeros([N,N]) 
        vert_knn_data[vert_array_sort, range_N] = pairwise_dists[vert_array_sort, range_N]

        intersection = np.zeros([N,N])
        intersection += (hor_knn_data==vert_knn_data)*hor_knn_data
        if (mutual):
            return intersection 
        else:
            return hor_knn_data+vert_knn_data - intersection
    
    def fully_connected_graph(self, data, sigma):
        """ returns a fully connected graph with gaussian similarity function """
        
        data_tup = np.stack((data.real, data.imag), axis=1)
        pairwise_sq_dists = squareform(pdist(data_tup, 'sqeuclidean'))
        G = sci.exp(-pairwise_sq_dists/(2*gamma**2))
        return G

    def unnormalized_laplacian(self, G):
        D = np.diag(np.sum(G, axis=1))
        return D-G

    def normalized_laplacian(self, G):
        """ sqrt(D)*l*sqrt(D) """
        N = G.shape[0] 
        sqrtD = np.sqrt(np.sum(G, axis=1))
        sqrtDinv = np.diag(np.divide(1.0, sqrtD, out=np.zeros(N), where=sqrtD!=0))
        return np.identity(N) - sqrtDinv.dot(G.dot(sqrtDinv)) 


