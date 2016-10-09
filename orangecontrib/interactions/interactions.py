import numpy as np
import scipy.sparse as sp
import Orange
from orangecontrib.interactions.utils import powerset, load_artificial_data


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

    def __repr__(self):
        msg = "Interaction beetween attributes " + str(self.a_name) + " and " + str(self.b_name) + ":\n"
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
        if self.sparse:  # If it is a sparse matrix, make it csc, because it enables fast column slicing operations
            self.data.X = sp.csc_matrix(self.data.X)
            self.data.Y = sp.csc_matrix(self.data.Y.reshape((self.m,1)))  # Seems like the easiest way to convert Y too
        self.info_gains = {self.data.domain.attributes[k].name: self.i(self.data.X[:, k], self.data.Y)
                           for k in range(self.n)}
        self.class_entropy = self.h(self.get_probs(self.data.Y))  # You will need this for relative information gain
        self.int_M_called = False

    def get_counts_sparse(self, x, with_counts=True):
        """Handles NaNs and counts the values in a 1D sparse array."""
        x_ = x.data[~np.isnan(x.data)]  # Getting just the non zero entries, excluding NaNs.
        no_non_zeros = len(x_)
        no_nans = len(x.data) - no_non_zeros
        no_zeros = x.shape[0] - no_nans - no_non_zeros
        uniques, counts = np.unique(x_, return_counts=True)  # count
        if no_zeros != 0:
            counts = np.concatenate((counts, [no_zeros]))  # adding the frequency of zeros
            uniques = np.concatenate((uniques, [0]))  # adding zero to uniques
        if with_counts:
            return counts, uniques
        else:
            return uniques

    def get_probs(self, *X):
        """
        Counts the frequencies of samples (rows) in a given n dimensional array ``*X`` and
        calculates probabilities with additive smoothing. Handles NaNs. Handles sparse arrays.

        Parameters
        ----------
        *X
            A sequence of 1D arrays (columns/attributes), can be sparse.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of probabilities.
        """
        no_att = len(X)
        if no_att == 1:
            if self.sparse:
                counts, uniques = self.get_counts_sparse(X[0])
                probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * len(uniques))  # additive smoothing
            else:
                x = X[0][~np.isnan(X[0])]  # remove NaNs
                uniques, counts = np.unique(x, return_counts=True)  # count
                probs = (counts + self.alpha) / (len(x) + self.alpha * len(uniques))  # additive smoothing
        else:
            if self.sparse:
                uniques = [self.get_counts_sparse(x, with_counts=False) for x in X]
                M = np.column_stack([x.toarray().flatten() for x in X])  # Get dense arrays and stack in a matrix
            else:
                uniques = [np.unique(x[~np.isnan(x)]) for x in X]  # Unique values for each attribute column, no NaNs.
                M = np.column_stack(X)  # Stack the columns in a matrix.
            k = np.prod([len(x) for x in uniques])  # Get the number of all possible combinations.
            M = M[~np.isnan(M).any(axis=1)]  # Remove samples that contain NaNs.
            m = M.shape[0]  # Number of samples remaining after NaNs have been removed.
            M_cont = np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * no_att)))
            # M_cont = M.view(M.dtype.descr * no_att)  # Using structured arrays is memory efficient, but a bit slower.
            _, counts = np.unique(M_cont, return_counts=True)  # Count the occurrences of joined attribute values.
            # The above line is the bottleneck. It accounts for 60% of time consumption of method get_probs
            counts = np.concatenate((counts, np.zeros(k - len(counts))))  # Add the zero frequencies
            # print(uniques.view(M.dtype).reshape(-1, no_att))  # Print uniques in a readable form.
            probs = (counts + self.alpha) / (m + self.alpha * k)  # Get probabilities (use additive smoothing).
        return probs

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

        return np.sum(-p*np.log2(p) if p > 0 else 0 for p in probs)

    def i(self, *X):
        """
        Computes information gain. 2 variables example: i(x,y) = h(x) + h(y) - h(x,y).

        Parameters
        ----------
        *X
            A sequence of 1D arrays (columns/attributes), can be sparse.

        Returns
        -------
        numpy.float64
            Information gain of attributes ``*X``.

        """
        if len(X) == 1:
            raise TypeError("Provide at least two arguments!")
        return np.sum([((-1) ** (len(subset) - 1)) * self.h(self.get_probs(*subset)) for subset in powerset(X)])

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

        a_name = self.data.domain.attributes[a].name
        b_name = self.data.domain.attributes[b].name
        ig_a = self.info_gains[a_name]  # We already have this info from initialization.
        ig_b = self.info_gains[b_name]
        if not total_rel_ig_ab:
            ig_ab = ig_a + ig_b - (self.class_entropy + self.h(self.get_probs(self.data.X[:, a], self.data.X[:, b]))) + \
                    self.h(self.get_probs(self.data.X[:, a], self.data.X[:, b], self.data.Y))
        else:
            ig_ab = ig_a + ig_b - total_rel_ig_ab * self.class_entropy
        inter = Interaction(a_name, b_name, ig_a, ig_b, ig_ab, self.class_entropy)
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
                int_M[k, j] = o.rel_total_ig_ab  # Store total information gain.
                int_M[j, k] = o.rel_total_ig_ab
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
