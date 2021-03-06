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


def load_mushrooms_data():
    shrooms_data = np.array(np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", delimiter=",", dtype=str))
    # Convert mushroom data from strings to integers
    names = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
            'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
             'spore-print-color', 'population', 'habitat']
    for i in range(len(shrooms_data[0, :])):
        u, ints = np.unique(shrooms_data[:, i], return_inverse=True)
        shrooms_data[:, i] = ints
    shrooms_data = shrooms_data.astype(np.float32)
    Y_shrooms = shrooms_data[:, 0]
    X_shrooms = shrooms_data[:, 1:]
    domain = Orange.data.Domain([Orange.data.DiscreteVariable(names[i-1], np.unique(X_shrooms[:, i-1])) for i in range(1,X_shrooms.shape[1]+1)],
                                Orange.data.DiscreteVariable("edible", np.unique(Y_shrooms)))
    data = Orange.data.Table(domain, X_shrooms, Y_shrooms)  # Make an Orange.Table object
    return data

# A wrapper to use with timeit module to time functions.
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped