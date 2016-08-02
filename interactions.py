import numpy as np
import Orange
from itertools import chain, combinations, product
from functools import reduce
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
    def __init__(self, data, disc_intervals=3, alpha=0.01):
        self.data = data
        self.n = len(self.data.domain.attributes)  # TODO: What is a better way to get this two numbers m and n
        self.m = len(self.data.X[:, 0])
        self.alpha = alpha # alpha is the smoothing parameter for additive (Laplace) smoothing of empirical probabilities
        # TODO: Check for sparse data
        discretizer = Orange.preprocess.Discretize() # Discretize continous attributes
        discretizer.method = Orange.preprocess.discretize.EqualFreq(disc_intervals)
        # TODO: should there be an option to choose the method
        # TODO: of dicscretization too? For now it is just equal frequences on three intervals.
        # TODO: What is the proper way to make this optional, input arguments when initializing Interactions class?
        self.data = discretizer(self.data)
        self.info_gains = self.get_information_gains()  # calculate info gain for all atrributes
        self.class_entropy = self.H(self.data.Y) #you will need this for relative information gain

    def H(self, *X):
        no_att = len(X)
        if no_att == 1:
            x = X[0][~np.isnan(X[0])] # remove NaNs
            uniques, counts = np.unique(x, return_counts=True)
            probs = (counts + 1) / (len(x) + len(uniques))
            return np.sum(-p*np.log2(p) for p in probs)
        else:
            uniques = [np.unique(x[~np.isnan(x)]) for x in X] # get unique values for each attribute column, don't count NaNs
            k = np.prod([len(x) for x in uniques]) # get the number of values in the cartesian product
            M = np.column_stack(X) # stack columns in a matrix
            M = M[~np.isnan(M).any(axis=1)] # remove samples that contain NaNs
            m = len(M) # number of samples remaining
            M_cont = np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * no_att))) # represent as contiguousarray
            # M_struct = M.view(M.dtype.descr * no_att)  # using structured arrays is memory efficient, but a little slower
            _, counts = np.unique(M_cont, return_counts=True) # count the ocurances of joined attribute values
            # print(uniques.view(M.dtype).reshape(-1, no_att)) # print uniques in a readble form
            probs = (counts + self.alpha) / (m + self.alpha*k) # get probabilities (additive smoothing)
            zero_p = self.alpha/(m + k) # the values that do not appear can have a non zero probability (additive smoothing)
            return np.sum(-p*np.log2(p) for p in probs) + ((k-len(counts))*(-zero_p*np.log2(zero_p)) if self.alpha !=0 else 0)# return entropy

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
        #TODO: Does DistMatrix need some metadata?
        int_M = np.zeros((self.n, self.n))
        for i in range(self.n): # Since this is a symetrix matrix we just compute the lower triangle and than copy
            for j in range(i+1): # TODO: I(X,X,Y) > I(X,Y) because of additive smoothing, but this is a kind of misinformation
# TODO: since an atrribute in combination with itself does not in fact provide more information, should diagonal elements be ommited then???
                int_M[i, j] = self.attribute_interactions(i, j).rel_ig_ab
        return Orange.misc.distmatrix.DistMatrix(int_M + int_M.T - np.diag(int_M.diagonal()))

def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])
    domain = Orange.data.Domain([Orange.data.DiscreteVariable("attribute1"), Orange.data.DiscreteVariable("attribute2")], Orange.data.DiscreteVariable("xor"))
    data = Orange.data.Table(domain, X, Y)
    return data

def load_artificial_data(no_att, no_samples, no_unique_values, no_classes, no_nans=False, no_class_nans=False):
    X = np.array([np.random.randint(no_unique_values, size=no_samples) for i in range(no_att)]).T.astype(np.float32)
    if no_nans:
        np.put(X, np.random.choice(range(no_samples*no_att), no_nans, replace=False), np.nan) #put in some nans
    Y = np.random.randint(no_classes, size=no_samples).astype(np.float32)
    if no_class_nans:
        np.put(Y, np.random.choice(range(no_samples), no_class_nans, replace=False), np.nan) #put in some nans
    data = Orange.data.Table(X, Y)
    return data

def load_mushrooms_data(no_samples=False, random_nans_no=False):
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
    domain = Orange.data.Domain([Orange.data.DiscreteVariable("attribute" + str(i)) for i in range(1,X_shrooms.shape[1]+1)],
                                Orange.data.DiscreteVariable("edible"))
    data = Orange.data.Table(domain, X_shrooms, Y_shrooms)  # Make an Orange.Table object
    return data

def test_H(data, a=0, b=1, alpha=0.01):
    inter = Interactions(data, alpha=alpha)
    print("Single entropy of attribute", data.domain.attributes[a], inter.H(data.X[:, a]))
    print("Single entropy of attribute", data.domain.attributes[b], inter.H(data.X[:, b]))
    print("Single entropy of class variable:", inter.H(data.Y))
    print("Joined entropy of attribute", data.domain.attributes[a], "and class variable:", inter.H(data.X[:, a], data.Y))
    print("Joined entropy of attribute", data.domain.attributes[b], "and class variable:", inter.H(data.X[:, b], data.Y))
    print("Joined entropy of attributes", data.domain.attributes[a], ",", data.domain.attributes[b],
          "and class variable:", inter.H(data.X[:, a], data.X[:, b], data.Y))

def test_I(data, a=0, b=1, alpha=0.01):
    #Test infromation gain function
    inter = Interactions(data, alpha)
    gain_0 = inter.I(data.X[:, a], data.Y)
    gain_1 = inter.I(data.X[:, b], data.Y)
    interaction_01 = inter.I(data.X[:, a], data.X[:, b], data.Y)
    print("Information gain for attribute", data.domain.attributes[a], ":", gain_0)
    print("Information gain for attribute", data.domain.attributes[b], ":",  gain_1)
    print("Interaction gain of atrributes", data.domain.attributes[a], "and", data.domain.attributes[b], ":", interaction_01)
    print("***************************************************************************")

def test_attribute_interactions(data, alpha=0.01):
    #Test attribute interactions method
    inter = Interactions(data, alpha=alpha)
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
    top = 1
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

def test_interaction_matrix(data, alpha=1):
    inter = Interactions(data, alpha=alpha)
    interaction_M = inter.interaction_matrix()
    print(interaction_M)
    # print(np.min(interaction_M))
    # print(np.max(interaction_M))
    # print(interaction_M.diagonal())
    # i, j = np.unravel_index(interaction_M.argmax(), interaction_M.shape)
    # print(i, j)

if __name__ == '__main__':
    # TODO: test correctnes of H and I
    # data = Orange.data.Table("lenses")  # Load discrete dataset
    # data = load_mushrooms_data(random_nans_no=500) # Load bigger discrete dataset
    data = Orange.data.Table("iris")  # load contionous dataset
    # TODO: test discretizer on lots of other datasets
    # data = load_mushrooms_data() # Load bigger discrete dataset
    # data = load_artificial_data(10, 500, 20, 2, 100, 10) # Load artificial data
    # data = load_xor_data()

    # inter = Interactions(data)

    # test_H(data)
    # test_I(data)
    test_attribute_interactions(data)
    # test_interaction_matrix(data)

    #CORRECTNES TESTING
    # print(inter.fast_H(data.X[:,5], data.X[:,11], data.Y))
    # inter.H(data.X[:,5])
    # inter.I(data.X[:,5], data.X[:,6])

    #TESTING NANS
    # np.random.seed(42)
    # a = list(np.random.randint(3, size=10))
    # a[np.random.randint(10,size=1)[0]] = np.nan
    # a = np.array(a)
    #
    # b = list(np.random.randint(3, size=10))
    # b[np.random.randint(10, size=1)[0]] = np.nan
    # b = np.array(b)
    #
    # c = list(np.random.randint(3, size=10))
    # c[np.random.randint(10, size=1)[0]] = np.nan
    # c = np.array(c)
    # print(a)
    # print(b)
    # print(c)
    # print(inter.H(a))
    # print(inter.H(a, b))
    # print(inter.H(a, b, c))

    #SPEED TESTING:
    # wrapped = wrapper(inter.H, data.X[:,0], data.X[:,1], data.Y)
    # print("H time:", timeit.timeit(wrapped, number=3))
    # ent = inter.H(data.X[:,0], data.X[:,1], data.Y)
    # print(ent)

    # wrapped = wrapper(test_interaction_matrix, data)
    # print("Time:", timeit.timeit(wrapped, number=3))



