import numpy as np
from Orange.data import Table

def H(l):
    """Takes a 1D np.array of discrete values representing an attribute and calculates its entropy."""
    #TODO: find out if categorical variables in orange data tables are always labeled with non negative integeres
    n = len(l) #number of instances in l
    keys, occurances = np.unique(l, return_counts=True) #calculate occurances of values in l
    k = len(keys) #number of unique values in l
    probs_laplace = [(o+1)/(n + k) for o in occurances] #probabilities with additive smoothing (alpha=1)
    prob_dist = dict(zip(keys, probs_laplace)) #present probability distribution in the format of a dictionary
    entropy = -sum(np.log2(probs_laplace))
    print("Unique values:", keys)
    print("Frequencies:", occurances)
    print("Probabilty distribution with additive smoothing:", prob_dist)
    print("Calculated entroby:", entropy)
    return entropy

class interactions:
    def __init__(self, data):
        self.data = data
        #TODO: Check for sparse data
        #TODO: Discretize continous attributes

    def attribute_interactions(self, a, b):
        return
        #TODO

if __name__ == '__main__':
    #test_dataset = np.array(np.loadtxt('./data/mushroom.txt', delimiter=",", dtype=str))
    d = Table("lenses")
    for i in range(len(d.domain.attributes)):
        print("Atribute name:", d.domain.attributes[i])
        H(d[i])
        print("***************************************************************************")

