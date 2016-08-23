import numpy as np
import scipy.sparse as sp
import Orange
import Orange.statistics.contingency
from interactions import *
import timeit


def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])
    domain = Orange.data.Domain([Orange.data.DiscreteVariable("attribute1"), Orange.data.DiscreteVariable("attribute2")], Orange.data.DiscreteVariable("xor"))
    data = Orange.data.Table(domain, X, Y)
    return data


def load_artificial_data(no_att, no_samples, no_unique_values, no_classes, no_nans=False, no_class_nans=False, sparse=False):
    X = np.array([np.random.randint(no_unique_values, size=no_samples) for i in range(no_att)]).T.astype(np.float32)
    if no_nans:
        np.put(X, np.random.choice(range(no_samples*no_att), no_nans, replace=False), np.nan) #put in some nans
    Y = np.random.randint(no_classes, size=no_samples).astype(np.float32)
    if no_class_nans:
        np.put(Y, np.random.choice(range(no_samples), no_class_nans, replace=False), np.nan) #put in some nans
    if sparse:
        np.put(X, np.random.choice(range(no_samples*no_att), sparse, replace=False), 0)  #make the X array sparse
        X = sp.csr_matrix(X)
    domain = Orange.data.Domain([Orange.data.DiscreteVariable("Attribute" + str(i), [str(j) for j in range(no_unique_values)])
                                 for i in range(1, X.shape[1] + 1)],
                                Orange.data.DiscreteVariable("Class_variable", [str(j) for j in range(no_classes)]))
    data = Orange.data.Table(domain, X, Y)
    return data


def load_mushrooms_data(no_samples=False, random_nans_no=False, sparse=False):
    shrooms_data = np.array(np.genfromtxt("./data/agaricus-lepiota.data", delimiter=",", dtype=str))
    # Convert mushroom data from strings to integers
    for i in range(len(shrooms_data[0, :])):
        u, ints = np.unique(shrooms_data[:, i], return_inverse=True)
        shrooms_data[:, i] = ints
    shrooms_data = shrooms_data.astype(np.float32)
    if random_nans_no:
        np.put(shrooms_data, np.random.choice(range(shrooms_data.shape[0]*shrooms_data.shape[1]), random_nans_no, replace=False), np.nan)
    # print(np.sum(np.isnan(shrooms_data)))
    if no_samples:
        #sample a smaller subset of mushrooms data
        np.random.shuffle(shrooms_data)
        shrooms_data = shrooms_data[:no_samples,:]
    Y_shrooms = shrooms_data[:, 0]
    X_shrooms = shrooms_data[:, 1:]
    if sparse:
        X_shrooms = sp.csr_matrix(X_shrooms)
    domain = Orange.data.Domain([Orange.data.DiscreteVariable("attribute" + str(i)) for i in range(1,X_shrooms.shape[1]+1)],
                                Orange.data.DiscreteVariable("edible"))
    data = Orange.data.Table(domain, X_shrooms, Y_shrooms)  # Make an Orange.Table object
    return data


# A wrapper to use with timeit module to time functions.
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == '__main__':
    # Example on how to use the class interaction:
    # d = Orange.data.Table("zoo") # Load  discrete dataset.
    # d = load_artificial_data(3, 1000, 30, 10, sparse=100) # Load sparse dataset
    d = load_mushrooms_data(sparse=True, random_nans_no=500) # Load bigger dataset.
    # d = Orange.data.Table("titanic")
    inter = Interactions(d) # Initialize Interactions object.
    # # Since info gain for single attributes is computed at initialization we can already look at it.
    # # To compute the interactions of all pairs of attributes we can use method interaction_matrix.
    # # We get a symmetric matrix, but the same info is also stored in a list internally.
    interacts_M = inter.interaction_matrix()
    # # We can get the 3 combinations that provide the most info about the class variable by using get_top_att
    best_total = inter.get_top_att(3, criteria="total")
    for i in best_total: # Interaction objects also print nicely.
        print(i)
        print("*****************************************************************")

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