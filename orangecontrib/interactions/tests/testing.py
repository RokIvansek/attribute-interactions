import numpy as np
import Orange
import pandas as pd
from orangecontrib.interactions.interactions import *
from orangecontrib.interactions.utils import *
import timeit
import time
from collections import Counter

def whole_process(s, a, top):
    d = load_artificial_data(s, a, 20, 5)
    inter = Interactions(d)
    inter.interaction_matrix()
    best = inter.get_top_att(top)
    return best

if __name__ == '__main__':
    # print("A script to mess around in and run tests.")

    #SPEED TESTING:

    m = 10000 # number of samples
    n = 20 # number of attributes
    v = 30 # number of unique attribute values
    c = 5 # number of unique class values

    d = load_artificial_data(n, m, v, c)
    inter = Interactions(d, alpha=0)

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

    start_0 = time.clock()
    probs_0 = pd.crosstab(inter.data.X[:, 0], [inter.data.X[:, 1], inter.data.X[:, 2]]).as_matrix()
    stop_0 = time.clock()
    print("panas crosstab time:", stop_0 - start_0)
    # print(probs_0)

    start_1 = time.clock()
    probs_1 = inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1], inter.data.X[:, 2])
    stop_1 = time.clock()
    print("np.unique:", stop_1 - start_1)
    # print(probs_1)

    # probs_0 = np.histogram2d(inter.data.X[:, 0], inter.data.X[:, 1], bins=[v, v], range=[[0, v], [0, v]])[0].flatten()/m
    # print(probs_0)
    #
    # probs_1 = inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1])
    # print(probs_1)

    # wrapped = wrapper(inter.interaction_matrix)
    # print("Time interaction_matrix:", timeit.timeit(wrapped, number=3) / 3)
    #
    # wrapped = wrapper(inter.get_top_att, 3)
    # print("Time get_top_att:", timeit.timeit(wrapped, number=3) / 3)

    #
    # print(inter.get_probs(inter.data.X[:, 0], inter.data.X[:, 1], inter.data.X[:, 2]))

    #
    # wrapped = wrapper(inter.get_probs, inter.data.X[:, 0], inter.data.X[:, 1], inter.data.Y)
    # print("Time get probs 3vars:", timeit.timeit(wrapped, number=3) / 3)
    #
    # wrapped = wrapper(Orange.statistics.contingency.Discrete, d, 0)
    # print("Time Orange contingency:", timeit.timeit(wrapped, number=3) / 3)