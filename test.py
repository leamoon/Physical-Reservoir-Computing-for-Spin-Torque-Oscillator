# the code applies another method to build a MTJ echo state network

import numpy as np
import matplotlib.pyplot as plt
import device_module as dm
import scipy.sparse
import scipy.sparse.linalg

np.random.seed(1000)


# a probability distribution function
def connection_reservoir(n_reservoir, density_reservoir, density_input):
    weight_state = scipy.sparse.rand(n_reservoir, n_reservoir, density=density_reservoir, format='coo')
    weight_in = scipy.sparse.rand(n_reservoir, 1, density=density_input, format='coo')
    weight_out = np.random.uniform(0, 1, (1, n_reservoir))
    return 0


if __name__ == '__main__':
    connection_reservoir(12, 0.6, 0.7)
