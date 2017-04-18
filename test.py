import numpy as np
def int_to_coord(x, n_bits):
    bin_rep = [int(y) for y in "{0:b}".format(x)]
    pad = n_bits - len(bin_rep)

    return tuple(np.pad(bin_rep, pad_width=(pad,0), mode='constant', constant_values=0))


qam16 = {
        (0, 0, 0, 0): 1.0/np.sqrt(2)*np.array([1, 1]),
        (0, 0, 0, 1): 1.0/np.sqrt(2)*np.array([2, 1]),
        (0, 0, 1, 0): 1.0/np.sqrt(2)*np.array([1, 2]),
        (0, 0, 1, 1): 1.0/np.sqrt(2)*np.array([2, 2]),
        (0, 1, 0, 0): 1.0/np.sqrt(2)*np.array([1, -1]),
        (0, 1, 0, 1): 1.0/np.sqrt(2)*np.array([1, -2]),
        (0, 1, 1, 0): 1.0/np.sqrt(2)*np.array([2, -1]),
        (0, 1, 1, 1): 1.0/np.sqrt(2)*np.array([2, -2]),
        (1, 0, 0, 0): 1.0/np.sqrt(2)*np.array([-1, 1]),
        (1, 0, 0, 1): 1.0/np.sqrt(2)*np.array([-1, 2]),
        (1, 0, 1, 0): 1.0/np.sqrt(2)*np.array([-2, 1]),
        (1, 0, 1, 1): 1.0/np.sqrt(2)*np.array([-2, 2]),
        (1, 1, 0, 0): 1.0/np.sqrt(2)*np.array([-1, -1]),
        (1, 1, 0, 1): 1.0/np.sqrt(2)*np.array([-2, -1]),
        (1, 1, 1, 0): 1.0/np.sqrt(2)*np.array([-1, -2]),
        (1, 1, 1, 1): 1.0/np.sqrt(2)*np.array([-2, -2])
    }

print(qam16[int_to_coord(6,4)])