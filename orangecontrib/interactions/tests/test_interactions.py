import numpy as np
import scipy.sparse as sp
import Orange
from orangecontrib.interactions.utils import *
from orangecontrib.interactions.interactions import *
import unittest


class ExampleTests(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)

    def test_xor(self):
        d = load_xor_data()
        inter = Interactions(d, alpha=0)
        int_M = inter.interaction_matrix().tolist()
        self.assertEqual(int_M, [[0.0, -1.0], [-1.0, 0.0]])