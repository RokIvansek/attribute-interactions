import numpy as np
import Orange
from orangecontrib.interactions.interactions import Interactions
from orangecontrib.interactions.utils import *
import timeit


if __name__ == '__main__':
    # print("A script to mess around in and run tests.")

    #SPEED TESTING:

    s = 100 # number of samples
    a = 10 # number of attributes
    u_a = 10 # number of unique attribute values
    u_c = 3 # number of unique class values

    d = load_artificial_data(s, a, u_a, u_c)
    inter = Interactions(d)

    print("Testing for", s, "samples,", a, "attributes:")

    # wrapped = wrapper(inter.interaction_matrix)
    # print("Time int matrix:", timeit.timeit(wrapped, number=3) / 3)

    # Testing for 100 samples, 10 attributes: (with np.unique)
    # Time int matrix: 2.495855596667146

    #TESTING GET PROBS

    wrapped = wrapper(inter.get_probs, inter.data.X[:, 0])
    print("Time get_probs:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(inter.get_probs_new, inter.data.X[:, 0])
    print("Time get_probs_:", timeit.timeit(wrapped, number=3) / 3)
