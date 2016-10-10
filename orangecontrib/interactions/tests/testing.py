import numpy as np
import Orange
from orangecontrib.interactions.interactions import *
from orangecontrib.interactions.utils import *
import timeit


if __name__ == '__main__':
    # print("A script to mess around in and run tests.")

    #SPEED TESTING:

    s = 100000 # number of samples
    a = 10 # number of attributes
    u_a = 10 # number of unique attribute values
    u_c = 3 # number of unique class values

    d = load_artificial_data(a, s, u_a, u_c, 1000, 100, sparse=1000)
    inter = Interactions(d)

    print("Testing for", s, "samples,", a, "attributes:")

    # wrapped = wrapper(inter.interaction_matrix)
    # print("Time int matrix:", timeit.timeit(wrapped, number=3) / 3)

    # Testing for 100 samples, 10 attributes: (with np.unique)
    # Time int matrix: 2.495855596667146

    #TESTING GET PROBS

    # 1 array
    # print(inter.data.X[:, 0])

    # wrapped = wrapper(inter.get_probs, inter.data.X[:, 0])
    # print("Time get_probs:", timeit.timeit(wrapped, number=3) / 3)
    # print(inter.get_probs(inter.data.X[:, 0]))
    # print(inter.h(inter.get_probs(inter.data.X[:, 0])))
    #
    # wrapped = wrapper(inter.get_probs_new, inter.data.domain.variables[0])
    # print("Time get_probs_new:", timeit.timeit(wrapped, number=3) / 3)
    # print(inter.get_probs_new(inter.data.domain.variables[0]))
    # print(inter.h(inter.get_probs_new(inter.data.domain.variables[0])))

    # 2 arrays

    # wrapped = wrapper(inter.get_probs, inter.data.X[:, 0], inter.data.Y)
    # print("Time get_probs:", timeit.timeit(wrapped, number=3) / 3)
    # print(inter.get_probs(inter.data.X[:, 0], inter.data.Y))
    # print(inter.h(inter.get_probs(inter.data.X[:, 0], inter.data.Y)))
    #
    # wrapped = wrapper(inter.get_probs_new, inter.data.domain.variables[0], inter.data.domain.variables[-1])
    # print("Time get_probs_new:", timeit.timeit(wrapped, number=3) / 3)
    # print(inter.get_probs_new(inter.data.domain.variables[0], inter.data.domain.variables[-1]))
    # print(inter.h(inter.get_probs_new(inter.data.domain.variables[0], inter.data.domain.variables[-1])))

    # 3 arrays

    wrapped = wrapper(inter.get_probs, inter.data.X[:, 0], inter.data.X[:, 1], inter.data.Y)
    print("Time get_probs:", timeit.timeit(wrapped, number=3) / 3)
    print(inter.h(inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1], inter.data.Y)))

    wrapped = wrapper(inter.get_probs_new, inter.data.domain.variables[0], inter.data.domain.variables[1],
                      inter.data.domain.variables[-1])
    print("Time get_probs_new:", timeit.timeit(wrapped, number=3) / 3)
    print(inter.get_probs_new(inter.data.domain.variables[0], inter.data.domain.variables[-1]))
    print(inter.h(inter.get_probs_new(inter.data.domain.variables[0], inter.data.domain.variables[1],
                                      inter.data.domain.variables[-1])))