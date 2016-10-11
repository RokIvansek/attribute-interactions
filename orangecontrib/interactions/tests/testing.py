import numpy as np
import Orange
from orangecontrib.interactions.interactions import *
from orangecontrib.interactions.utils import *
import timeit


if __name__ == '__main__':
    # print("A script to mess around in and run tests.")

    #SPEED TESTING:

    s = 10 # number of samples
    a = 10 # number of attributes
    u_a = 3 # number of unique attribute values
    u_c = 2 # number of unique class values
    nans_att = 20
    nans_class = 3
    sparse = False

    d = load_artificial_data(a, s, u_a, u_c, nans_att, nans_class, sparse)
    print(d)
    inter = Interactions(d)

    print("Testing for:\n",
          s, "samples,\n",
          a, "attributes, \n",
          u_a, "unique_attribute_values, \n",
          u_c, "unique_class_values, \n",
          nans_att, "nans in attributes, \n",
          nans_class, "nans in class, \n",
          "sparse:", sparse, ":")

    # wrapped = wrapper(inter.interaction_matrix)
    # print("Time int matrix:", timeit.timeit(wrapped, number=3) / 3)

    # Testing for 100 samples, 10 attributes: (with np.unique)
    # Time int matrix: 2.495855596667146

    # Testing for 100 samples, 10 attributes: (with orange contingency and bincount)
    # Time int matrix: 0.08377752533321352

    #GET PROBS

    # print(np.sum(inter.get_probs(inter.data.domain.variables[0], inter.data.domain.variables[1], inter.data.domain.variables[-1])))
    print(np.sum(inter.get_probs(inter.data.domain.variables[0], inter.data.domain.variables[-1])))
    # print(np.sum(inter.get_probs(inter.data.domain.variables[0])))

