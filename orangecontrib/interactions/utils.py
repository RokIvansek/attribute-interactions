import Orange
import numpy as np
import scipy as sp
from itertools import chain, combinations


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))  # without empty subset


def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])
    domain = Orange.data.Domain([Orange.data.DiscreteVariable("Attribute" + str(i), [str(j) for j in range(2)])
                                 for i in range(1, X.shape[1] + 1)],
                                Orange.data.DiscreteVariable("xor", [str(j) for j in range(2)]))
    data = Orange.data.Table(domain, X, Y)
    return data


def load_artificial_data(no_att, no_samples, no_unique_values, no_classes=False, no_nans=False, no_class_nans=False, sparse=False):
    X = np.array([np.random.randint(no_unique_values, size=no_samples) for i in range(no_att)]).T.astype(np.float32)
    if no_nans:
        np.put(X, np.random.choice(range(no_samples*no_att), no_nans, replace=False), np.nan) #put in some nans
    if sparse:
        np.put(X, np.random.choice(range(no_samples*no_att), sparse, replace=False), 0)  #make the X array sparse
        X = sp.sparse.csr_matrix(X)
    if no_classes:
        Y = np.random.randint(no_classes, size=no_samples).astype(np.float32)
        if no_class_nans:
            np.put(Y, np.random.choice(range(no_samples), no_class_nans, replace=False), np.nan) #put in some nans
        domain = Orange.data.Domain([Orange.data.DiscreteVariable("Attribute" + str(i), [str(j) for j in range(no_unique_values)])
                                     for i in range(1, X.shape[1] + 1)],
                                    Orange.data.DiscreteVariable("Class_variable", [str(j) for j in range(no_classes)]))
        data = Orange.data.Table(domain, X, Y)
    else:
        domain = Orange.data.Domain([Orange.data.DiscreteVariable("Attribute" + str(i), [str(j) for j in range(no_unique_values)])
                                     for i in range(1, X.shape[1] + 1)])
        data = Orange.data.Table(domain, X)
    return data


def load_mushrooms_data(no_samples=False, random_nans_no=False, sparse=False):
    shrooms_data = np.array(np.genfromtxt("../datasets/agaricus-lepiota.data", delimiter=",", dtype=str))
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