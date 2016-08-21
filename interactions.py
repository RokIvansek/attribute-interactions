import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, issparse
import Orange
from itertools import chain, combinations
import timeit


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))  # without empty subset


class Interaction:
    """
    A class that stores all the values needed to compute the interaction chart.
    """
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
        # TODO: Should this print abolute info gains too? They are not really informative.
        msg = "Interaction beetween attributes " + str(self.a_name) + " and " + str(self.b_name) + ".\n"
        msg += "Relative info gain for attribute " + str(self.a_name) + ": " + str(self.rel_ig_a) + "\n"
        msg += "Relative info gain for attribute " + str(self.b_name) + ": " + str(self.rel_ig_b) + "\n"
        msg += "Relative info gain for both attributes together: " + str(self.rel_ig_ab) + "\n"
        msg += "Total relative info gain from attributes and their combination: " + str(self.rel_total_ig_ab)
        return msg

    def __repr__(self):  # TODO: Do I need to overwrite this method too?
        msg = "Interaction beetween attributes " + str(self.a_name) + " and " + str(self.b_name) + ".\n"
        msg += "Relative info gain for attribute " + str(self.a_name) + ": " + str(self.rel_ig_a) + "\n"
        msg += "Relative info gain for attribute " + str(self.b_name) + ": " + str(self.rel_ig_b) + "\n"
        msg += "Relative info gain for both attributes together: " + str(self.rel_ig_ab) + "\n"
        msg += "Total relative info gain from attributes and their combination: " + str(self.rel_total_ig_ab)
        return msg


class Interactions:
    """

    A class for computing attribute interactions using the concept of Shannon entropy from information theory.
    It works on discrete datasets - continuous attributes/classes are discretized using equal frequencies
    discretization.
    """
    def __init__(self, data, disc_intervals=3, alpha=0.01): # TODO: add preprocesor object
        """

        :param data: Orange data table.
        :param disc_intervals: Number of intervals produced when discretizing continuous attributes.
        :param alpha: Additive (Laplace) smoothing parameter.
        """
        self.data = data
        self.n = self.data.X.shape[1]  # TODO: What is a better way to get this two numbers m and n?
        self.m = self.data.X.shape[0]
        self.alpha = alpha
        # TODO: Check for sparse data
        discretizer = Orange.preprocess.Discretize()
        discretizer.method = Orange.preprocess.discretize.EqualFreq(disc_intervals)
        # TODO: Should there be an option to choose the method
        # TODO: of dicscretization too? For now it is just equal frequencies on three intervals.
        # TODO: What is the proper way to make this optional, input arguments when initializing Interactions class?
        self.data = discretizer(self.data)  # Discretize continous attributes
        self.sparse = issparse(self.data.X)  # Check for sparsity
        if self.sparse: # If it is a sparse matrix, make it csc, because it enables fast column slicing operations
            self.data.X = csc_matrix(self.data.X)
        # self.info_gains = {self.data.domain.attributes[i]: self.i(self.data.X[:, i], self.data.Y)
        #                    for i in range(self.n)}
        # self.class_entropy = self.h(self.get_probs(self.data.Y))  # You will need this for relative information gain
        self.all_pairs = [] # Here we will store the Interaction objects for all possible pairs of attributes.

    def get_counts_sparse(self, x, with_counts=True):
        """Handle NaNs and count the values in a 1D sparse array."""
        x_ = x.data[~np.isnan(x.data)]  # Getting just the non zero entries, excluding NaNs.
        no_non_zeros = len(x_)
        no_nans = len(x.data) - no_non_zeros
        no_zeros = x.shape[0] - no_nans - no_non_zeros
        uniques, counts = np.unique(x_, return_counts=True)  # count
        if no_zeros != 0:
            counts = np.concatenate((counts, [no_zeros]))  # adding the frequency of zeros
            uniques = np.concatenate((uniques, [0]))  # adding zero to uniques
        if with_counts:
            return (counts, uniques)
        else:
            return uniques

    def count_rows_sparse(self, X):
        """Handle NaNs and return the frequencies of rows in a sparse matrix X"""
        # TODO: Try finding a faster routine for counting rows. Converting to dense and then counting
        # TODO: is 3 times slower than the routine for non sparse data (tested on the same data).
        return

    def get_probs(self, *X):
        """
        Counts the frequencies of samples (rows) in a given n dimensional array X and
        calculates probabilities with additive smoothing. Handles NaNs. Handles sparse arrays.

        :param X: A sequence of 1D arrays (columns/attributes), can be sparse.
        :return: Probabilities
        """

        no_att = len(X)
        if no_att == 1:
            if issparse(X[0]):
                counts, uniques = self.get_counts_sparse(X[0])
                probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * len(uniques))  # additive smoothing
            else:
                x = X[0][~np.isnan(X[0])]  # remove NaNs
                uniques, counts = np.unique(x, return_counts=True)  # count
                probs = (counts + self.alpha) / (len(x) + self.alpha * len(uniques))  # additive smoothing
        else:
            if issparse(X[0]):
                uniques = [self.get_counts_sparse(x, with_counts=False) for x in X]
                M = np.column_stack((tuple([x.toarray() for x in X])))  # Get dense arrays and stack in a matrix
            else:
                uniques = [np.unique(x[~np.isnan(x)]) for x in X]  # Unique values for each attribute column, no NaNs.
                M = np.column_stack((X))  # Stack the columns in a matrix.
            k = np.prod([len(x) for x in uniques])  # Get the number of all possible combinations.
            M = M[~np.isnan(M).any(axis=1)]  # Remove samples that contain NaNs.
            m = M.shape[0]  # Number of samples remaining after NaNs have been removed.
            M_cont = np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * no_att)))
            # M_cont = M.view(M.dtype.descr * no_att)  # Using structured arrays is memory efficient, but a bit slower.
            _, counts = np.unique(M_cont, return_counts=True)  # Count the occurrences of joined attribute values.
            counts = np.concatenate((counts, np.zeros(k - len(counts)))) # Add the zero frequencies
            # print(uniques.view(M.dtype).reshape(-1, no_att))  # Print uniques in a readable form.
            probs = (counts + self.alpha) / (m + self.alpha * k)  # Get probabilities (use additive smoothing).
        return probs

    def h(self, probs):
        """
        Computes single/joined entropy from probabilities.

        :param probs: A 1-dim array containing probabilities.
        :return: Single/joined entropy.
        """

        return np.sum(-p*np.log2(p) for p in probs)

    def i(self, *X):
        """Computes information gain. 2 variables example: I(X;Y) = H(X) + H(Y) - H(XY)"""
        return np.sum([((-1) ** (len(subset) - 1)) * self.h(self.get_probs(*subset)) for subset in powerset(X)])

    def attribute_interactions(self, a, b):
        """

        Calculates attribute interaction between attributes a and b i.e. I(a, b, y), where y is a class variable.

        :param a: Attribute name or index or anything that Orange can use to access the attributes.
        :param b: Attribute name or index or anything that Orange can use to access the attributes.
        :return: Object of type Interaction.
        """

        # TODO: How to access atrribute values (columns) if name of attribute is given?
        # TODO: For now suppose a and b are integers. If they are not, first get integers or else below code won't work.
        a_name = self.data.domain.attributes[a]
        b_name = self.data.domain.attributes[b]
        ig_a = self.info_gains[a_name]  # We already have this info from initialization.
        ig_b = self.info_gains[b_name]
        # ig_ab = self.i(self.data.X[:, a], self.data.X[:, b],  self.data.Y) # Instead of computing everything again, we
        # can use what we already have and just add what we need I(A:B:Y) = I(A:Y) + I(B:Y) - H(Y) - H(A:B) + H(A:B:Y)
        ig_ab = ig_a + ig_b - (self.class_entropy + self.h(self.get_probs(self.data.X[:, a], self.data.X[:, b]))) + \
                self.h(self.get_probs(self.data.X[:, a], self.data.X[:, b], self.data.Y))
        inter = Interaction(a_name, b_name, ig_a, ig_b, ig_ab, self.class_entropy)
        return inter

    def interaction_matrix(self):
        """Computes the relative total information gain for all possible couples of attributes."""
        # TODO: Since this method iterates trough all possible pairs of attributes, would it be wise to calculate
        # TODO: a running sort of some kind to later access the most informative pairs without having to sort again?
        int_M = np.zeros((self.n, self.n))
        for i in range(self.n):  # Since this is a symetric matrix we just compute the lower triangle and then copy
            for j in range(i+1):  # TODO: i(X,X,Y) > i(X,Y) because of additive smoothing, but this is a kind of
        # TODO: misinformation since an atrribute in combination with itself does not in fact provide more information.
        # TODO: Should diagonal elements be ommited then???
                o = self.attribute_interactions(i, j)
                int_M[i, j] = o.rel_ig_ab  # Store actual interaction info
                self.all_pairs.append(o) # Stores the entire Interaction object in a list
        return Orange.misc.distmatrix.DistMatrix(int_M + int_M.T - np.diag(int_M.diagonal()))

    def get_top_att(self, n, criteria=["total", "interaction"]):
        """
        Returns the Interaction objects of n most informative pairs of attributes.
        For this to work, interaction_matrix must be called first.
        """
        if criteria == "total":
            self.all_pairs.sort(key=lambda x: x.rel_total_ig_ab, reverse=True)
            return self.all_pairs[:n]
        if criteria == "interaction":
            self.all_pairs.sort(key=lambda x: x.rel_ig_ab)
            return self.all_pairs[:n]

# THE ACTUAL LIBRARY ENDS HERE
# ****************************


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
        X = csr_matrix(X)
    data = Orange.data.Table(X, Y)
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
        X_shrooms = csr_matrix(X_shrooms)
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
    # d = load_artificial_data(100, 1000, 20, 5, sparse=50000) # Load sparse dataset
    # d = load_mushrooms_data(sparse=True) # Load bigger dataset.
    # inter = Interactions(d) # Initialize Interactions object.
    # Since info gain for single attributes is computed at initialization we can already look at it.
    # To compute the interactions of all pairs of attributes we can use method interaction_matrix.
    # We get a symmetric matrix, but the same info is also stored in a list internally.
    # interacts_M = inter.interaction_matrix()
    # # We can get the 3 combinations that provide the most info about the class variable by using get_top_att
    # best_total = inter.get_top_att(3, criteria="total")
    # for i in best_total: # Interaction objects also print nicely.
    #     print(i)
    #     print("*****************************************************************")

    #TEST get_probs FOR SPARSE TABLE WITH NANS
    d = load_mushrooms_data(sparse=True)  # Load bigger dataset.
    inter = Interactions(d)  # Initialize Interactions object.

    r = 2
    s = 100
    nans = 100

    x1 = np.random.randint(r, size=s).astype(np.float32)
    x1 = x1.reshape((s, 1))
    np.put(x1, np.random.choice(range(len(x1)), size=nans, replace=False), np.nan)
    x1_sparse = csr_matrix(x1)

    x2 = np.random.randint(r, size=s).astype(np.float32)
    x2 = x2.reshape((s, 1))
    np.put(x2, np.random.choice(range(len(x1)), size=nans, replace=False), np.nan)
    x2_sparse = csr_matrix(x2)

    x3 = np.random.randint(r, size=s).astype(np.float32)
    x3 = x3.reshape((s, 1))
    np.put(x3, np.random.choice(range(len(x1)), size=nans, replace=False), np.nan)
    x3_sparse = csr_matrix(x3)

    # print(np.sort(inter.get_probs(x1, x2, x3)))
    # print(np.sort(inter.get_probs(x1_sparse, x2_sparse, x3_sparse)))

    #SPEED TESTING:
    # d = load_artificial_data(100, 1000, 50, 10)
    # inter = Interactions(d, alpha=0)

    #testin sparse routines
    wrapped = wrapper(inter.get_probs, x1, x2, x3)
    print("time non sparse:", timeit.timeit(wrapped, number=3) / 3)

    wrapped = wrapper(inter.get_probs, x1_sparse, x2_sparse, x3_sparse)
    print("time sparse:", timeit.timeit(wrapped, number=3) / 3)

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



