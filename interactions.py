import numpy as np
from Orange.data import Table
from Orange.misc import distmatrix
from sklearn.utils.extmath import cartesian
from functools import reduce
from itertools import chain, combinations
import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1)) #added one since we dont want empty subsets

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
        self.n = len(self.data.domain.attributes)  # TODO: What is a better way to get this two numbers m and n
        self.m = len(self.data.X[:, 0])
        # TODO: Check for sparse data
        # TODO: Discretize continous attributes
        self.info_gains = self.get_information_gains()  # calculate info gain for all atrributes
        self.class_entropy = self.H(self.data.Y) #you will need this for relative information gain

    def H(self, *X):
        no_att = len(X)
        if no_att == 1:
            uniques, counts = np.unique(X[0], return_counts=True)
            probs = (counts + 1) / (self.m + len(uniques))
            return np.sum(-p*np.log2(p) for p in probs)
        else:
            uniques = [np.unique(x) for x in X] # get unique values for each attribute column
            k = np.prod([len(x) for x in uniques]) # get the number of values in the cartesian product
            M = np.column_stack(X) # stack columns in a matrix
            M = np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * no_att))) # represent as contiguousarray
            # M = M.view(M.dtype.descr * no_att)  # using structured arrays is memory efficient, but a little slower
            _, counts = np.unique(M, return_counts=True) # count the ocurances of joined attribute values
            probs = (counts + 1) / (self.m + k) # get probabilities (additive smoothing)
            zero_p = 1/(self.m + k) # the values that do not appear have a non zero probability (additive smoothing)
            return np.sum(-p*np.log2(p) for p in probs) + (k-len(counts))*(-zero_p*np.log2(zero_p)) # return entropy

    def I(self, *X):
        return np.sum([((-1) ** (len(subset) - 1)) * self.H(*subset) for subset in powerset(X)])

    def get_information_gains(self):
        # TODO: What if there is more class variables?
        info_gains = {self.data.domain.attributes[i]: self.I(self.data.X[:, i], self.data.Y) for i in range(self.n)}
        # TODO: What is the right way to access columns of data in Orange data table?
        return info_gains

    def attribute_interactions(self, a, b):
        """
        :param a: attribute name or index or anything that Orange can use to access the attributes
        :param b: attribute name or index or anything that Orange can use to access the attributes
        :return: object of type interaction containing all the information for drawing the interaction chart
        """

        #TODO: How to access atrribute values (columns) if name of attribute is given?
        #For now suppose a and b are integers
        ig_a = self.I(self.data.X[:, a], self.data.Y)
        ig_b = self.I(self.data.X[:, b], self.data.Y)
        ig_ab = self.I(self.data.X[:, a], self.data.X[:, b],  self.data.Y)
        a_name = self.data.domain.attributes[a]
        b_name = self.data.domain.attributes[b]
        inter = Interaction(a_name, b_name, ig_a, ig_b, ig_ab, self.class_entropy)
        return inter

    def interaction_matrix(self):
        #TODO: Does this matrix contain Interaction classes as values or some specific value like relative info gain of both attributes?
        #TODO: Is there a more efficient way to do this since this is a symetric matrix?
        #TODO: Does DistMatrix need some metadata?
        return distmatrix.DistMatrix(np.array([[self.attribute_interactions(a, b).rel_ig_ab for a in range(self.n)] for b in range(self.n)]))

def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])
    data = Table(X, Y)
    return data

def load_artificial_data(no_att, no_samples, no_unique_values, no_classes):
    X = np.array([np.random.randint(no_unique_values, size=no_samples) for i in range(no_att)]).T
    Y = np.random.randint(no_classes, size=no_samples)
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

def test_H(data, a=0, b=1):
    inter = Interactions(data)
    print("Single entropy of attribute", data.domain.attributes[a], inter.H(data.X[:, a]))
    print("Single entropy of attribute", data.domain.attributes[b], inter.H(data.X[:, b]))
    print("Single entropy of class variable:", inter.H(data.Y))
    print("Joined entropy of attribute", data.domain.attributes[a], "and class variable:", inter.H(data.X[:, a], data.Y))
    print("Joined entropy of attribute", data.domain.attributes[b], "and class variable:", inter.H(data.X[:, b], data.Y))
    print("Joined entropy of attributes", data.domain.attributes[a], ",", data.domain.attributes[b],
          "and class variable:", inter.H(data.X[:, a], data.X[:, b], data.Y))

def test_I(data, a=0, b=1):
    #Test infromation gain function
    inter = Interactions(data)
    gain_0 = inter.I(data.X[:, a], data.Y)
    gain_1 = inter.I(data.X[:, b], data.Y)
    interaction_01 = inter.I(data.X[:, a], data.X[:, b], data.Y)
    print("Information gain for attribute", data.domain.attributes[a], ":", gain_0)
    print("Information gain for attribute", data.domain.attributes[b], ":",  gain_1)
    print("Interaction gain of atrributes", data.domain.attributes[a], "and", data.domain.attributes[b], ":", interaction_01)
    print("***************************************************************************")

def test_attribute_interactions(data):
    #Test attribute interactions method
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

def test_interaction_matrix(data):
    inter = Interactions(data)
    interaction_M = inter.interaction_matrix()
    print(interaction_M.shape)

if __name__ == '__main__':
    # TODO: test correctnes of H and I
    # data = Table("lenses")  # Load discrete dataset
    # data = load_mushrooms_data() # Load bigger discrete dataset
    data = load_artificial_data(1000, 50000, 50, 10) # Load artificial data
    inter = Interactions(data)
    # data = load_xor_data()
    # test_H(data)
    # test_I(data)
    # test_attribute_interactions(data)
    # test_interaction_matrix(data)

    #GENERATE SOME RANDOM DATA
    # np.random.seed(42)
    # a = np.random.randint(50, size=10000)
    # b = np.random.randint(50, size=10000)
    # c = np.random.randint(20, size=10000)

    #CORRECTNES TESTING
    # print(inter.fast_H(data.X[:,5], data.X[:,11], data.Y))
    # inter.H(data.X[:,5])
    # inter.I(data.X[:,5], data.X[:,6])

    #SPEED TESTING:
    # wrapped = wrapper(inter.H, data.X[:,0], data.X[:,1], data.Y)
    # print("H time:", timeit.timeit(wrapped, number=3))
    # ent = inter.H(data.X[:,0], data.X[:,1], data.Y)
    # print(ent)

    # wrapped = wrapper(test_interaction_matrix, data)
    # print("Time:", timeit.timeit(wrapped, number=3))



