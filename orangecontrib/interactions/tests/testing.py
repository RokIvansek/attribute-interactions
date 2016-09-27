import numpy as np
import Orange
from orangecontrib.interactions.interactions import *
from orangecontrib.interactions.utils import *
import timeit
import time

def whole_process(s, a, top):
    d = load_artificial_data(s, a, 20, 5)
    inter = Interactions(d)
    inter.interaction_matrix()
    best = inter.get_top_att(top)
    return best

if __name__ == '__main__':
    # print("A script to mess around in and run tests.")

    #SPEED TESTING:

    m = 10 # number of samples
    n = 2 # number of attributes
    v = 3 # number of unique attribute values
    c = 2 # number of unique class values

    d = load_artificial_data(n, m, v, c, 10)
    print(d)

    inter = Interactions(d, alpha=0)
    print(inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1]))

    # print(inter.data.Y.shape)
    print(Orange.statistics.util.contingency(inter.data.X[:, 0], inter.data.X[:, 1]))
    # print(Orange.statistics.distribution.Discrete(d, 0))

    print("Testing for", m, "samples,", n, "attributes:")
    # print(d)

    # start_0 = time.clock()
    # _, uni_0 = np.unique(inter.data.X[:, 0], return_counts=True)
    # stop_0 = time.clock()
    # print("np.unique time:", stop_0 - start_0)
    # print(uni_0)
    #
    # start_1 = time.clock()
    # uni_1 = np.bincount(inter.data.X[:, 0].astype(int))
    # stop_1 = time.clock()
    # print("bincount time:", stop_1 - start_1)
    # print(uni_1)

    # start_1 = time.clock()
    # probs_1 = inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1], inter.data.X[:, 2])
    # stop_1 = time.clock()
    # print("np.unique:", stop_1 - start_1)
    # print(probs_1)

    # probs_0 = np.histogram2d(inter.data.X[:, 0], inter.data.X[:, 1], bins=[v, v], range=[[0, v], [0, v]])[0].flatten()/m
    # print(probs_0)
    #
    # probs_1 = inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1])
    # print(probs_1)

    wrapped = wrapper(inter.get_probs, inter.data.X[:, 0])
    print("Time get_probs:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(Orange.statistics.util.bincount, inter.data.X[:, 0])
    print("Time orange contingency:", timeit.timeit(wrapped, number=3) / 3)

    #
    # print(inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1], inter.data.X[:, 2]))

    #
    # wrapped = wrapper(inter.get_probs, inter.data.X[:, 0], inter.data.X[:, 1], inter.data.Y)
    # print("Time get probs 3vars:", timeit.timeit(wrapped, number=3) / 3)
    #
    # wrapped = wrapper(Orange.statistics.contingency.Discrete, d, 0)
    # print("Time Orange contingency:", timeit.timeit(wrapped, number=3) / 3)