import numpy as np
import scipy.sparse as sp
import Orange
from orangecontrib.interactions.utils import *
from orangecontrib.interactions.interactions import *
import unittest


class ExampleTests(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)

    def test_xor(self):
        d = load_xor_data()
        inter = Interactions(d, alpha=0)
        int_M = inter.interaction_matrix().tolist()
        self.assertEqual(int_M, [[0.0, -1.0], [-1.0, 0.0]])

if __name__ == '__main__':
    print("This is a call from test_interactions.py")
    #TEST get_probs FOR SPARSE TABLE WITH NANS
    # d = load_mushrooms_data(sparse=True)  # Load bigger dataset.
    # inter = Interactions(d)  # Initialize Interactions object.
    #
    # r = 10
    # s = 100000
    # nans = 10
    #
    # x1 = np.random.randint(r, size=s).astype(np.float32)
    # x1 = x1.reshape((s, 1))
    # np.put(x1, np.random.choice(range(len(x1)), size=nans, replace=False), np.nan)
    # x1_sparse = sp.csr_matrix(x1)
    #
    # x2 = np.random.randint(r, size=s).astype(np.float32)
    # x2 = x2.reshape((s, 1))
    # np.put(x2, np.random.choice(range(len(x1)), size=nans, replace=False), np.nan)
    # x2_sparse = sp.csr_matrix(x2)
    #
    # x3 = np.random.randint(r, size=s).astype(np.float32)
    # x3 = x3.reshape((s, 1))
    # np.put(x3, np.random.choice(range(len(x1)), size=nans, replace=False), np.nan)
    # x3_sparse = sp.csr_matrix(x3)

    # print(np.sort(inter.get_probs(x1, x2, x3)))
    # print(np.sort(inter.get_probs(x1_sparse, x2_sparse, x3_sparse)))

    #SPEED TESTING:

    # d = Orange.data.Table("zoo")
    # inter = Interactions(d)

    # print(Orange.statistics.contingency.Discrete(d, 0))
    # print(inter.get_probs(inter.data.X[:,0], inter.data.Y))

    #testing speed of contingency table
    # wrapped = wrapper(inter.get_probs, inter.data.X[:,0], inter.data.Y)
    # print("time my contingency:", timeit.timeit(wrapped, number=3) / 3)
    #
    # wrapped = wrapper(Orange.statistics.contingency.Discrete, d, 0)
    # print("time Orange contingency:", timeit.timeit(wrapped, number=3) / 3)

    #testing sparse routines
    # wrapped = wrapper(inter.get_probs, x1, x2, x3)
    # print("time non sparse:", timeit.timeit(wrapped, number=3) / 3)
    #
    # wrapped = wrapper(inter.get_probs, x1_sparse, x2_sparse, x3_sparse)
    # print("time sparse:", timeit.timeit(wrapped, number=3) / 3)

    #testing i
    # wrapped = wrapper(inter.h, d.X[:, 0])
    # ent = inter.h(d.X[:, 0])
    # print(ent)
    # print("time:", timeit.timeit(wrapped, number=3)/3)

    # wrapped = wrapper(inter.i, d.X[:, 0], d.X[:, 1])
    # print("time:", timeit.timeit(wrapped, number=3)/3)
    #
    # wrapped = wrapper(inter.i, d.X[:,0], d.X[:,1], d.Y)
    # print("time:", timeit.timeit(wrapped, number=3)/3)

    #testing attribute_interactions
    # wrapped = wrapper(inter.attribute_interactions, 0, 1)
    # print("time:", timeit.timeit(wrapped, number=3)/3)

    #testing interaction matrix
    # wrapped = wrapper(inter.interaction_matrix)
    # print("time:", timeit.timeit(wrapped, number=3) / 3)

    # ent = inter.h(d.X[:,0], d.X[:,1], d.Y)
    # print(ent)

    # wrapped = wrapper(test_interaction_matrix, d)
    # print("Time:", timeit.timeit(wrapped, number=3))