import numpy as np
from Orange.data import Table
from sklearn.utils.extmath import cartesian

def H(*rand_vars, return_prob_dist=False):
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
    for key in info_gains:
        print(key, "info gain:", info_gains[key])
    return info_gains

class Interaction:
    def __init__(self, abs_ig_a, abs_ig_b, abs_ig_ab, class_ent):
        self.abs_ig_a = abs_ig_a
        self.abs_ig_b = abs_ig_b
        self.abs_ig_ab = abs_ig_ab
        self.rel_ig_a = self.abs_ig_a/class_ent
        self.rel_ig_b = self.abs_ig_b/class_ent
        self.rel_ig_ab = self.abs_ig_ab/class_ent

    def __str__(self):
        return "Write this method so it will print multiple lines."

    def __repr__(self):
        return "Write this method so it will print multiple lines."


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
        inter = Interaction(ig_a, ig_b, ig_ab, self.class_entropy)
        return inter
        #TODO

def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])
    data = Table(X, Y)
    return data

def load_mushrooms_data():
    shrooms_data = np.array(np.genfromtxt("./data/agaricus-lepiota.data", delimiter=",", dtype=str))
    # Convert mushroom data from strings to integers
    for i in range(len(shrooms_data[0, :])):
        u, ints = np.unique(shrooms_data[:, i], return_inverse=True)
        shrooms_data[:, i] = ints
    #sample a smaller subset of mushrooms data
    np.random.shuffle(shrooms_data)
    shrooms_data = shrooms_data[:100,:]
    Y_shrooms = shrooms_data[:, 0]
    X_shrooms = shrooms_data[:, 1:]
    data = Table(X_shrooms, Y_shrooms)  # Make and Orange.Table object, without domain added
    # it thinks attributes are countinous but for this test it doesn't really matter
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
    print("All information gains of individual attributes:")
    inter = Interactions(data)
    #Testing class Interaction and relative info gain calculation
    a = 0
    b = 3
    chart_info = inter.attribute_interactions(a, b)
    print(chart_info.rel_ig_a)
    print(chart_info.rel_ig_b)
    print(chart_info.rel_ig_ab)

if __name__ == '__main__':
    # TODO: check correctnes of H and I, something is off
    # data = Table("lenses")  # Load discrete dataset
    # data = load_mushrooms_data() # Load bigger discrete dataset
    data = load_xor_data()
    print(data.X)
    print(data.Y)
    test_H(data)
    test_I(data)
    # test_Interactions(data)
    # a = np.random.randint(5, size=1000)
    # b = np.random.randint(5, size=1000)
    # c = cartesian((a,b))
    # c_ = c.astype(str)



