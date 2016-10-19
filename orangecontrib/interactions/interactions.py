import numpy as np
import scipy.sparse as sp
import bottleneck as bn
import Orange
from orangecontrib.interactions.utils import powerset
from _functools import reduce
from operator import add, mul



class Interaction:
    """
    A class that stores all the interaction values between two attributes.

    Parameters
    ----------
    a_name : str
        Name of the attribute.
    b_name : str
        Name of the attribute.
    abs_ig_a :
        Absolute info gain of attribute a.
    abs_ig_b :
        Absolute info gain of attribute b.
    abs_ig_ab :
        Absolute info gain of attributes a and b combined.
    class_ent :
        Entropy of the class attribute.
    """
    def __init__(self, var_a, var_b, abs_ig_a, abs_ig_b, abs_ig_ab, class_ent):
        self.var_a = var_a
        self.var_b = var_b
        self.abs_ig_a = abs_ig_a
        self.abs_ig_b = abs_ig_b
        self.abs_ig_ab = abs_ig_ab
        self.abs_total_ig_ab = abs_ig_a + abs_ig_b - abs_ig_ab
        self.rel_ig_a = self.abs_ig_a/class_ent
        self.rel_ig_b = self.abs_ig_b/class_ent
        self.rel_ig_ab = self.abs_ig_ab/class_ent
        self.rel_total_ig_ab = self.abs_total_ig_ab/class_ent

    def __repr__(self):
        msg = "Interaction beetween attributes " + str(self.var_a.name) + " and " + str(self.var_b.name) + ":\n"
        msg += "Relative info gain for attribute " + str(self.var_a.name) + ": " + str(self.rel_ig_a) + "\n"
        msg += "Relative info gain for attribute " + str(self.var_b.name) + ": " + str(self.rel_ig_b) + "\n"
        msg += "Relative info gain for both attributes together: " + str(self.rel_ig_ab) + "\n"
        msg += "Total relative info gain from attributes and their combination: " + str(self.rel_total_ig_ab)
        return msg


class Interactions:
    """

    A class for computing attribute interactions using the concept of Shannon entropy from information theory.
    It works on discrete datasets - continuous attributes/classes are discretized using equal frequencies
    discretization.

    Parameters
    ----------
    data : Orange.data.Table
        Orange data table.
    disc_method : Orange.preprocess.Discretization
        A method for discretizing continuous attributes.
    alpha : float
        Additive (Laplace) smoothing parameter (default: 0.01).
    """
    def __init__(self, data, disc_method=Orange.preprocess.discretize.EqualFreq(3), alpha=0.01):
        if data.domain.class_var is None:
            raise AttributeError("Classless data set!")
        self.data = data
        self.n = self.data.X.shape[1]
        self.m = self.data.X.shape[0]
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be an element of interval [0, 1]")
        self.alpha = alpha
        discretizer = Orange.preprocess.Discretize()
        discretizer.method = disc_method  # TODO: Is this OK? Method as input argument?
        self.data = discretizer(self.data)
        self.sparse = sp.issparse(self.data.X)  # Check for sparsity
        # TODO: Check if this conversion to csc is still necessary in the new get_probs
        if self.sparse:  # If it is a sparse matrix, make it csc, because it enables fast column slicing operations
            self.data.X = sp.csc_matrix(self.data.X)
            self.data.Y = sp.csc_matrix(self.data.Y.reshape((self.m,1)))  # Seems like the easiest way to convert Y too
        self.info_gains = {self.data.domain.attributes[k].name:
                           self.i(self.data.domain.variables[k], self.data.domain.variables[-1])
                           for k in range(self.n)}
        self.class_entropy = self.h(self.get_probs(self.data.domain.variables[-1]))  # You will need this for relative information gain
        self.int_M_called = False

    def freq_counts(self, arrs, lens):
        """
        Calculates frequencies of samples.

        Parameters
        ----------
        arrs
            A sequence of arrays.
        lens
            A sequence of number of distinct values in arrays.
        Returns
        -------
        numpy.ndarray
            A 1D numpy array of frequencies.

        """
        no_nans = reduce(np.logical_and, [~np.isnan(a) if bn.anynan(a) else np.ones(self.m).astype(bool) for a in arrs])
        combined = reduce(add, [arrs[i][no_nans]*reduce(mul, lens[:i]) for i in range(1, len(arrs))], arrs[0][no_nans])
        return np.bincount(combined.astype(np.int32, copy=False), minlength=reduce(mul, lens)).astype(float)

    def get_probs(self, *vars):
        """
        Counts the frequencies of samples of given variables ``*vars`` and
        calculates probabilities with additive smoothing.

        Parameters
        ----------
        *vars
            A sequence of Orange discrete variables.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of probabilities.
        """
        freqs = self.freq_counts([self.data.get_column_view(v)[0] for v in vars], [len(v.values) for v in vars])
        k = np.prod([len(v.values) for v in vars])
        return (freqs + self.alpha) / (np.sum(freqs) + self.alpha*k)

    def h(self, probs):
        """
        Entropy of a given distribution.

        Parameters
        ----------
        probs
            A 1-dim array of probabilities.

        Returns
        -------
        numpy.float64
            Single/joined entropy value.

        """

        return np.sum(-p*np.log2(p) if p > 0 else 0 for p in np.nditer(probs))

    def i(self, *vars):
        """
        Computes information gain. 2 variables example: i(x,y) = h(x) + h(y) - h(x,y).

        Parameters
        ----------
        *X
            A sequence of Orange variables.

        Returns
        -------
        numpy.float64
            Information gain of attributes ``*vars``.

        """
        if len(vars) == 1:
            raise TypeError("Provide at least two arguments!")
        return np.sum([((-1) ** (len(subset) - 1)) * self.h(self.get_probs(*subset)) for subset in powerset(vars)])

    def attribute_interactions(self, a, b, total_rel_ig_ab=None):
        """
        If not given, computes the absolute total info gain for attributes a and b. Generates an Interaction object.

        Parameters
        ----------
        a
            Attribute index.
        b
            Attribute index.
        total_rel_ig_ab
            Total relative info gain of attributes.

        Returns
        -------
        Interaction
            Object of type Interaction.

        """
        var_a = self.data.domain.variables[a]
        var_b = self.data.domain.variables[b]
        ig_a = self.info_gains[var_a.name]
        ig_b = self.info_gains[var_b.name]
        if not total_rel_ig_ab:
            ig_ab = ig_a + ig_b - (self.class_entropy + self.h(self.get_probs(var_a, var_b))) + \
                    self.h(self.get_probs(var_a, var_b, self.data.domain.variables[-1]))
        else:
            ig_ab = ig_a + ig_b - total_rel_ig_ab * self.class_entropy
        inter = Interaction(var_a, var_b, ig_a, ig_b, ig_ab, self.class_entropy)
        return inter

    def interaction_matrix(self):
        """
        Computes a symetric matrix containing the relative total information gain for all possible pairs of attributes.
        """

        self.int_M_called = True
        int_M = np.zeros((self.n, self.n))
        for k in range(self.n):
            for j in range(k+1):
                o = self.attribute_interactions(k, j)
                int_M[k, j] = o.rel_total_ig_ab  # Store total information gain
                int_M[j, k] = o.rel_total_ig_ab  # TODO: Maybe storing interactions too is not a bad idea
                # TODO: We can than easily sort either by total gain or by positive interaction
        for k in range(self.n):
            int_M[k, k] = self.info_gains[self.data.domain.attributes[k].name]
        self.int_matrix = Orange.misc.distmatrix.DistMatrix(int_M)

    def get_top_att(self, n):
        """
        Computes the Interaction objects for n most informative pairs of attributes.
        For this to work, ``interaction_matrix`` must be called first.
        It uses a partial sort and then a full sort on the remaining n elements to get the indices of attributes.


        Parameters
        ----------
        n
            The number of attribute pairs we wish to get.

        Returns
        -------
        list
            A list of Interaction objects.
        """
        if not self.int_M_called:
            raise IndexError("Call interaction_matrix first!")
        flat_indices = np.argpartition(np.tril(-self.int_matrix, -1).ravel(), n - 1)[:n]
        # TODO: Consider using the partial sort from the bottleneck module for faster sorting
        row_indices, col_indices = np.unravel_index(flat_indices, self.int_matrix.shape)
        min_elements_order = np.argsort(-self.int_matrix[row_indices, col_indices])
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]
        return [self.attribute_interactions(row_indices[k], col_indices[k],
                                            self.int_matrix[row_indices[k], col_indices[k]]) for k in range(n)]


if __name__ == '__main__':
    # Example on how to use the class interaction:
    d = Orange.data.Table("zoo") # Load  discrete dataset.
    # d = Orange.data.Table("iris") # Load continuous dataset.
    inter = Interactions(d) # Initialize Interactions object.
    # To compute the interactions of all pairs of attributes we can use method interaction_matrix.
    inter.interaction_matrix()
    # We can get the 3 combinations that provide the most info about the class variable by using get_top_att
    best_total = inter.get_top_att(3)
    for i in best_total: # Interaction objects also print nicely.
        print(i)
        print("*****************************************************************")