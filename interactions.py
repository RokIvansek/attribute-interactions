import numpy as np
from Orange.data import Table
from sklearn.utils.extmath import cartesian
from functools import reduce
import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def H(*X):
    n = len(X[0])
    uniques = [np.unique(x) for x in X]
    k = np.prod([len(x) for x in uniques])
    return np.sum(-p * np.log2(p) if p > 0 else 0 for p in
        ((np.sum(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes)))) + 1)/(n + k)
            for classes in cartesian(uniques)))

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
    test_H(data)
    test_I(data)
    test_Interactions(data)

    #GENERATE SOME RANDOM DATA
    # np.random.seed(42)
    # a = np.random.randint(50, size=10000)
    # b = np.random.randint(50, size=10000)
    # c = np.random.randint(50, size=10000)
    # a_ = np.random.randint(50, size=10000)
    # b_ = np.random.randint(50, size=10000)
    # c_ = np.random.randint(50, size=10000)

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

    #SPEED TESTING:
    # wrapped = wrapper(H, a_, b_)
    # print("H (fast one) time:", timeit.timeit(wrapped, number=10))
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
    # wrapped = wrapper(H, a_, b_, c_)
    # print("H time:", timeit.timeit(wrapped, number=10))
    # ent = H(a, b, c)
    # print(ent)
    # wrapped = wrapper(H_, a_, b_, c_)
    # print("H_ time:", timeit.timeit(wrapped, number=10))
    # ent = H_(a, b, c)
    # print(ent)



