import numpy as np
from Orange.data import Table
from sklearn.utils.extmath import cartesian
import timeit
import itertools
from functools import reduce

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

# This is the fixed entropy function.
# Probabilities are now correctly calculated and additive smoothing was added. It is still super fast. :)
def h_0(*X):
    n = len(X[0])
    H = 0
    uniques = [set(x) for x in X]
    ds = [len(x) for x in uniques] #needed for additive smoothing
    # print(ks)
    for classes in itertools.product(*uniques):
        p = np.prod([(np.sum(predictions == c)+1)/(n+d) for predictions, c, d in zip(X, classes, ds)])
        #Results will differ because of the mistake generated when multiplying probabilities (a lot of floats)
        H += -p * np.log2(p) if p > 0 else 0
    return H

def h_1(*X):
    n = len(X[0])
    uniques = [set(x) for x in X]
    ds = [len(x) for x in uniques]  # needed for additive smoothing
    return np.sum(-p * np.log2(p) if p > 0 else 0 for p in
                  (np.prod([(np.sum(predictions == c)+1)/(n+d) for predictions, c, d in zip(X, classes, ds)])
                   for classes in itertools.product(*uniques)))

def h_2(*X):
    n = len(X[0])
    # print(n_insctances)
    H = 0
    # print([set(x) for x in X])
    for classes in itertools.product(*[set(x) for x in X]):
        # print(classes)
        # print(v)
        # for predictions, c in zip(X, classes):
        #     print(predictions, c)
        v = reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes)))
        # for predictions, c in zip(X, classes):
        #     v = np.logical_and(v, predictions == c)
        p = np.mean(v)
        H += -p * np.log2(p) if p > 0 else 0
    return H

#Stole this from http://blog.biolab.si/2012/06/15/computing-joint-entropy-in-python/
#It performs great.
#TODO: Make sure the probabilities use additive smoothing (alpha = 1) t.i. Laplace probabilities
def H(*X):
    return np.sum(-p * np.log2(p) if p > 0 else 0 for p in
        (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
            for classes in itertools.product(*[set(x) for x in X])))

def h_3(*rand_vars, return_prob_dist=False):
    """
    :param rand_vars: discrete random variables as a tuple of 1-D arrays

    :return: entropy
    If one random variable X is given, entropy H(X) is calcuated.
    If two random variables X and Y are given, joint entropy H(XY) is calcuated.
    If three random variables X, Y and Z are given, joint entropy H(XYZ) is calculated.
    """

    #TODO: find out if categorical variables in orange data tables are always labeled with non negative integers
    v = len(rand_vars)
    if v == 1:
        #R = ["{}".format(i) for i in rand_vars[0]]
        R = rand_vars[0].astype(str) #turning this to strings looks like a bad idea
    elif v == 2:
        X, Y = rand_vars
        cart = cartesian((X,Y)).astype(str)
        #R = ["{}_{}".format(i, j) for i, j in cartesian((X, Y))] #reformat the cartesian product array
        R = list(map(lambda row: row[0] + row[1], cart))
        #into a list of strings so that np.unique will work on it
    elif v == 3:
        X, Y, Z = rand_vars
        cart = cartesian((X, Y, Z)).astype(str)
        #R = ["{}_{}_{}".format(i, j, k) for i, j, k in cartesian((X, Y, Z))]
        R = list(map(lambda row: row[0] + row[1] + row[2], cart))
    else:
        # TODO: How to properly raise errors/exceptions?
        print("Provide at least 1 and up to 3 random variables.")
        return
    n = len(R) #number of instances in R
    keys, counts = np.unique(R, return_counts=True) #count occurances of unique values
    k = len(keys) #number of unique values in R
    probs_laplace = [(c+1)/(n + k) for c in counts] #probabilities with additive smoothing (alpha=1)
    # probs_laplace = [c/n for c in counts]  # probabilities with additive smoothing (alpha=1)
    prob_dist = dict(zip(keys, probs_laplace)) #present probability distribution in the format of a dictionary
    entropy = -sum(probs_laplace*np.log2(probs_laplace)) #calculate entropy
    # print("Unique values:", keys)
    # print("Frequencies:", counts)
    # print("Probabilty distribution with additive smoothing:", prob_dist)
    # print("Calculated entroby:", entropy)
    if return_prob_dist:
        return entropy, prob_dist
    else:
        return entropy

#This version of entropy seems to work the fastest!
def h_4(*X):
    n = len(X[0])
    counts = [np.bincount(x) for x in X] #count occurances
    probs = [(cnt + 1)/(n + len(cnt))for cnt in counts] #apply additive smoothing
    H = 0
    for ps in cartesian(probs): #cartesian seems to work faster than itertools.product
        joint_prob = np.prod(ps)
        H += -joint_prob*np.log2(joint_prob)
    return H

def h_5(*X):
    n = len(X[0])
    counts = [np.bincount(x) for x in X] #count occurances
    probs = [(cnt + 1)/(n + len(cnt))for cnt in counts] #apply additive smoothing
    H = 0
    # for ps in cartesian(probs): #Not sure whether to use cartesian or itertool.product here
    for ps in itertools.product(*probs):
        joint_prob = np.prod(ps)
        H += -joint_prob*np.log2(joint_prob)
    return H


def I(*rand_vars):
    v = len(rand_vars)
    if v == 2:
        X, Y = rand_vars
        info_gain = H(X) + H(Y) - H(X,Y)
    elif v == 3:
        X, Y, Z = rand_vars
        info_gain = H(X) + H(Y) + H(Z) -(H(X, Y) + H(X, Z) + H(Y, Z)) + H(X, Y, Z)
    else:
        #TODO: How to properly raise errors/exceptions?
        print("Provide 2 or 3 random variables")
        return
    # print("Information gain:", info_gain)
    return info_gain

def get_information_gains(data):
    #TODO: What if there is more class variables?
    n = len(data.domain.attributes)
    info_gains = {data.domain.attributes[i]: I(data.X[:,i], data.Y) for i in range(n)}
    # TODO: What is the right way to access columns of data in Orange data table?
    return info_gains

class Interaction:
    def __init__(self, a_name, b_name, abs_ig_a, abs_ig_b, abs_ig_ab, class_ent):
        self.a_name = a_name
        self.b_name = b_name
        self.abs_ig_a = abs_ig_a
        self.abs_ig_b = abs_ig_b
        self.abs_ig_ab = abs_ig_ab
        self.abs_total_ig_ab = abs_ig_a + abs_ig_b - abs_ig_ab
        self.rel_ig_a = self.abs_ig_a/class_ent
        self.rel_ig_b = self.abs_ig_b/class_ent
        self.rel_ig_ab = self.abs_ig_ab/class_ent
        self.rel_total_ig_ab = self.abs_total_ig_ab/class_ent

    def __str__(self):
        #TODO: Should this print abolute info gains too? They are not really informative.
        msg = "Object representing interaction beetween attributes " + str(self.a_name) + " and " + str(self.b_name) + ".\n"
        msg += "Relative info gain for attribute " + str(self.a_name) + ": " + str(self.rel_ig_a) + "\n"
        msg += "Relative info gain for attribute " + str(self.b_name) + ": " + str(self.rel_ig_b) + "\n"
        msg += "Relative info gain for both attributes together: " + str(self.rel_ig_ab) + "\n"
        msg += "Total relative info gain from attributes and their combination: " + str(self.rel_total_ig_ab)
        return msg

    def __repr__(self):
        return "Do I need to overwrite this too?"


class Interactions:
    def __init__(self, data):
        self.data = data
        #TODO: Check for sparse data
        #TODO: Discretize continous attributes
        self.info_gains = get_information_gains(self.data) #calculate info gain for all atrributes
        self.class_entropy = H(self.data.Y) #you will need this for relative information gain

    def attribute_interactions(self, a, b):
        """
        :param a: attribute name or index or anything that Orange can use to access the attributes
        :param b: attribute name or index or anything that Orange can use to access the attributes
        :return: object of type interaction containing all the information for drawing the interaction chart
        """

        #TODO: How to access atrribute values (columns) if name of attribute is given?
        #For now suppose a and b are integers
        ig_a = I(self.data.X[:, a], self.data.Y)
        ig_b = I(self.data.X[:, b], self.data.Y)
        ig_ab = I(self.data.X[:, a], self.data.X[:, b],  self.data.Y)
        a_name = self.data.domain.attributes[a]
        b_name = self.data.domain.attributes[b]
        inter = Interaction(a_name, b_name, ig_a, ig_b, ig_ab, self.class_entropy)
        return inter
        #TODO

def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])
    data = Table(X, Y)
    return data

def load_mushrooms_data(no_samples=False):
    shrooms_data = np.array(np.genfromtxt("./data/agaricus-lepiota.data", delimiter=",", dtype=str))
    # Convert mushroom data from strings to integers
    for i in range(len(shrooms_data[0, :])):
        u, ints = np.unique(shrooms_data[:, i], return_inverse=True)
        shrooms_data[:, i] = ints
    if no_samples:
        #sample a smaller subset of mushrooms data
        np.random.shuffle(shrooms_data)
        shrooms_data = shrooms_data[:no_samples,:]
    Y_shrooms = shrooms_data[:, 0]
    X_shrooms = shrooms_data[:, 1:]
    data = Table(X_shrooms, Y_shrooms)  # Make and Orange.Table object, without domain added
    # it thinks attributes are countinous but for this test it doesn't really matter, to fix this add domain
    return data

def test_H(data):
    #Test entropy function
    # for i in range(len(data.domain.attributes)):
    #     print("Atribute name:", data.domain.attributes[i])
    #     entropy = H(data.X[:,i])
    #     print(entropy)
    a = 0
    b = 1
    print("Single entropy of attribute", data.domain.attributes[a], H(data.X[:, a]))
    print("Single entropy of attribute", data.domain.attributes[b], H(data.X[:, b]))
    print("Single entropy of class variable:", H(data.Y))
    print("Joined entropy of attribute", data.domain.attributes[a], "and class variable:", H(data.X[:, a], data.Y))
    print("Joined entropy of attribute", data.domain.attributes[b], "and class variable:", H(data.X[:, b], data.Y))
    print("Joined entropy of attributes", data.domain.attributes[a], ",", data.domain.attributes[b],
          "and class variable:", H(data.X[:, a], data.X[:, b], data.Y))

def test_I(data):
    #Test infromation gain function
    gain_0 = I(data.X[:, 0], data.Y)
    gain_1 = I(data.X[:, 1], data.Y)
    interaction_01 = I(data.X[:, 0], data.X[:, 1], data.Y)
    print("Information gain for attribute", data.domain.attributes[0], ":", gain_0)
    print("Information gain for attribute", data.domain.attributes[1], ":",  gain_1)
    print("Interaction gain of atrributes", data.domain.attributes[0], "and", data.domain.attributes[1], ":", interaction_01)
    print("***************************************************************************")

def test_Interactions(data):
    #Test interactions class
    inter = Interactions(data)
    print("-------------------------------------------------------------------------")
    print("All absolute information gains of individual attributes:")
    print("-------------------------------------------------------------------------")
    for key in inter.info_gains:
        print(key, "abs info gain:", inter.info_gains[key])
    print("-------------------------------------------------------------------------")
    print("All relative information gains of individual attributes:")
    print("-------------------------------------------------------------------------")
    for key in inter.info_gains:
        print(key, "rel info gain:", inter.info_gains[key]/inter.class_entropy)
    #Print out info gain for all of the pairs of attributes
    charts = []
    for a in range(len(data.domain.attributes)):
        for b in range(a+1, len(data.domain.attributes)):
            chart_info = inter.attribute_interactions(a, b)
            charts.append(chart_info)
    charts.sort(key=lambda x: x.rel_total_ig_ab, reverse=True)
    top = 5
    print("-------------------------------------------------------------------------")
    print("Top", top, "attribute combinations with highest total relative info gain:")
    print("-------------------------------------------------------------------------")
    for i in range(top):
        chart_info = charts[i]
        print(chart_info)
        print("****************************************************************************")
    charts.sort(key=lambda x: x.rel_ig_ab)
    print("-------------------------------------------------------------------------")
    print("Top", top, "attribute combinations with lowest relative info gain:")
    print("-------------------------------------------------------------------------")
    for i in range(top):
        chart_info = charts[i]
        print(chart_info)
        print("****************************************************************************")
    charts.sort(key=lambda x: x.rel_ig_ab, reverse=True)
    print("-------------------------------------------------------------------------")
    print("Top", top, "attribute combinations with highest relative info gain:")
    print("-------------------------------------------------------------------------")
    for i in range(top):
        chart_info = charts[i]
        print(chart_info)
        print("****************************************************************************")

if __name__ == '__main__':
    # TODO: test correctnes of H and I
    # data = Table("lenses")  # Load discrete dataset
    data = load_mushrooms_data() # Load bigger discrete dataset
    # data = load_xor_data()
    # test_H(data)
    # test_I(data)
    # test_Interactions(data)

    np.random.seed(42)
    a = np.random.randint(50, size=10000)
    b = np.random.randint(50, size=10000)
    c = np.random.randint(50, size=10000)
    a_ = np.random.randint(50, size=10000)
    b_ = np.random.randint(50, size=10000)
    c_ = np.random.randint(50, size=10000)

    # ent_0, prob_0 = true_ent(a, b)
    # ent_1, prob = H_slow(a, b, return_prob_dist=True)
    # prob_1 = []
    # for key in prob:
    #     prob_1.append(prob[key])
    # print(ent_0, ent_1)
    # prob_0.sort()
    # prob_1.sort()
    # for i in range(len(prob_0)):
    #     print(prob_0[i], prob_1[i])

    wrapped = wrapper(H, a_, b_)
    print("H (fast one) time:", timeit.timeit(wrapped, number=10))
    # ent = H(a, b, c)
    # print(ent)
    # wrapped = wrapper(h_0, a_, b_)
    # print("h_0 time:", timeit.timeit(wrapped, number=10))
    # ent = h_0(a, b, c)
    # print(ent)
    # wrapped = wrapper(h_1, a_, b_)
    # print("h_1 time:", timeit.timeit(wrapped, number=10))
    # ent = h_1(a, b, c)
    # print(ent)
    # wrapped = wrapper(h_2, a, b)
    # print("h_2 time:", timeit.timeit(wrapped, number=10))
    # ent = h_2(a, b, c)
    # print(ent)
    # wrapped = wrapper(h_3, a, b)
    # print("h_3 (correct one) time:", timeit.timeit(wrapped, number=10))
    # ent = h_3(a, b, c)
    # print(ent)
    wrapped = wrapper(h_4, a_, b_)
    print("h_4 time:", timeit.timeit(wrapped, number=10))
    # ent = h_4(a, b, c)
    # print(ent)
    wrapped = wrapper(h_5, a_, b_)
    print("h_5 time:", timeit.timeit(wrapped, number=10))
    # ent = h_5(a, b, c)
    # print(ent)



