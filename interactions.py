import numpy as np
from Orange.data import Table
from sklearn.utils.extmath import cartesian

def H(*rand_vars, return_prob_dist=False):
    """Takes discrete random variables as a tuple of 1-D arrays and calculates their entropy.

    If one random variable X is given, entropy H(X) is calcuated.
    If two random variables X and Y are given, joint entropy H(XY) is calcuated.
    If three random variables X, Y and Z are given, joint entropy H(XYZ) is calculated.
    """

    #TODO: find out if categorical variables in orange data tables are always labeled with non negative integers
    v = len(rand_vars)
    if v == 1:
        R = ["{}".format(i) for i in rand_vars[0]]
    elif v == 2:
        X, Y = rand_vars
        R = ["{}_{}".format(i, j) for i, j in cartesian((X, Y))] #reformat the cartesian product array
        #into a list of strings so that np.unique will work on it
    elif v == 3:
        X, Y, Z = rand_vars
        R = ["{}_{}_{}".format(i, j, k) for i, j, k in cartesian((X, Y, Z))]
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
    n = len(data.domain.attributes)
    info_gains = {data.domain.attributes[i]: I(data.X[:,i], data.Y) for i in range(n)}
    for key in info_gains:
        print(key, "info gain:", info_gains[key])
    return info_gains

class Interactions:
    def __init__(self, data):
        self.data = data
        #TODO: Check for sparse data
        #TODO: Discretize continous attributes
        self.info_gains = get_information_gains(self.data)

    def attribute_interactions(self, a, b):
        return
        #TODO

if __name__ == '__main__':
    #Test entropy function
    d = Table("lenses") #Load discrete dataset
    # for i in range(len(d.domain.attributes)):
    #     print("Atribute name:", d.domain.attributes[i])
    #     H(d[:,i])
    #     print("***************************************************************************")
    # H(d.X[:, 0], d.X[:, 1], d.Y)
    #Test infromation gain function
    gain_0 = I(d.X[:, 0], d.Y)
    gain_1 = I(d.X[:, 1], d.Y)
    interaction_01 = I(d.X[:, 0], d.X[:, 1], d.Y)
    print("Information gain for attribute", d.domain.attributes[0], ":", gain_0)
    print("Information gain for attribute", d.domain.attributes[1], ":",  gain_1)
    print("Interaction gain of atrributes", d.domain.attributes[0], "and", d.domain.attributes[1], ":", interaction_01)
    print("***************************************************************************")
    #Test interactions class
    print("All information gains of individual attributes for lenses dataset:")
    inter = Interactions(d)
    #TODO: Test on a bigget dataset. Maybe mushrooms or some other big categorical dataset.


