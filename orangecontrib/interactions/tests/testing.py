import numpy as np
import Orange
from orangecontrib.interactions.interactions import *
from orangecontrib.interactions.utils import *
import timeit


if __name__ == '__main__':
    # print("A script to mess around in and run tests.")

    #SPEED TESTING:

    s = 100000 # number of samples
    a = 50 # number of attributes
    u_a = 10 # number of unique attribute values
    u_c = 5 # number of unique class values
    nans_att = 1000
    nans_class = 10
    sparse =  1000000

    d = load_artificial_data(a, s, u_a, u_c, nans_att, nans_class, sparse)
    # print(d)
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

    # print(inter.h(inter.get_probs(inter.data.domain.variables[0], inter.data.domain.variables[1], inter.data.domain.variables[-1])))
    # print(inter.h(inter.get_probs(inter.data.domain.variables[0], inter.data.domain.variables[-1])))
    # print(inter.h(inter.get_probs(inter.data.domain.variables[0])))
    # print('**********************************************************************')
    # print(inter.h(inter.get_probs_old(inter.data.domain.variables[0], inter.data.domain.variables[1], inter.data.domain.variables[-1])))
    # print(inter.h(inter.get_probs_old(inter.data.domain.variables[0], inter.data.domain.variables[-1])))
    # print(inter.h(inter.get_probs_old(inter.data.domain.variables[0])))


    #FREQ_COUNTS
    var0 = inter.data.domain.variables[0]
    var1 = inter.data.domain.variables[1]
    var2 = inter.data.domain.variables[-1]
    a0 = inter.data.get_column_view(var0)[0]
    a1 = inter.data.get_column_view(var1)[0]
    a2 = inter.data.get_column_view(var2)[0]
    # print(a0)
    # print(a1)
    # print(a2)
    # print(inter.freq_counts([a0, a1, a2], [len(var0.values), len(var1.values), len(var2.values)]))

    wrapped = wrapper(inter.get_probs, var0)
    print("Time get probs 1D:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(inter.get_probs, var0, var1)
    print("Time get probs 2D:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(inter.get_probs, var0, var1, var2)
    print("Time get probs 3D:", timeit.timeit(wrapped, number=3) / 3)
