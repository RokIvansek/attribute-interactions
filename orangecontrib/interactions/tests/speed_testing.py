import numpy as np
import Orange
from orangecontrib.interactions.interactions import *
from orangecontrib.interactions.utils import *
import timeit

if __name__ == '__main__':
    #CORRECTNES TESTING



    #SPEED TESTING:

    m = 1000000

    d = load_artificial_data(5, m, 30, 10, 5000, 100)
    inter = Interactions(d, alpha=0)

    print(inter.data.domain.variables[0].values)

    print("Testing for", m, "samples:")

    wrapped = wrapper(inter.get_probs, inter.data.X[:, 0])
    print("Time get probs 1var:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(inter.get_probs, inter.data.X[:,0], inter.data.Y)
    print("Time get probs 2vars:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(inter.get_probs, inter.data.X[:, 0], inter.data.X[:, 1], inter.data.Y)
    print("Time get probs 3vars:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(Orange.statistics.contingency.Discrete, d, 0)
    print("Time Orange contingency:", timeit.timeit(wrapped, number=3) / 3)

    # Testing
    # for 1000000 samples:
    #     Time
    #     get
    #     probs
    #     1
    #     var: 0.05119527799994709
    # Time
    # get
    # probs
    # 2
    # vars: 0.42950214466660935
    # Time
    # get
    # probs
    # 3
    # vars: 0.6087319959998846
    # Time
    # Orange
    # contingency: 0.07875381766659