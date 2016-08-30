import numpy as np
import scipy.sparse as sp
import Orange
from orangecontrib.interactions.utils import *
from orangecontrib.interactions.interactions import *
import unittest


class InteractionsTests(unittest.TestCase):
    def test_classless(self):
        d = load_artificial_data(3, 10, 2)
        self.assertRaises(AttributeError, Interactions, d)

    def test_discretization_method_type(self):
        d = Orange.data.Table('iris')
        self.assertRaises(TypeError, Interactions, d, disc_method="a string")
        self.assertRaises(TypeError, Interactions, d, disc_method=5)

    def test_alpha_value(self):
        d = load_artificial_data(3, 10, 2, 2)
        self.assertRaises(ValueError, Interactions, d, alpha=-1)

    def test_probs(self):
        d = load_artificial_data(3, 10, 2, 2)  # Dense without nans
        inter = Interactions(d)
        self.assertTrue(1-1e-10 <= np.sum(inter.get_probs(inter.data.X[:,0])) <= 1+1e-10)
        self.assertTrue(1-1e-10 <= np.sum(inter.get_probs(inter.data.X[:,0], inter.data.Y)) <= 1+1e-10)

    def test_probs_nans(self):
        d = load_artificial_data(3, 10, 2, 2, 10, 3)  # Dense with nans
        inter = Interactions(d)
        self.assertTrue(1 - 1e-10 <= np.sum(inter.get_probs(inter.data.X[:, 0])) <= 1 + 1e-10)
        self.assertTrue(1 - 1e-10 <= np.sum(inter.get_probs(inter.data.X[:, 0], inter.data.Y)) <= 1 + 1e-10)

    def test_probs_sparse(self):
        d = load_artificial_data(3, 10, 2, 2, sparse=10)  # Sparse
        inter = Interactions(d)
        self.assertTrue(1 - 1e-10 <= np.sum(inter.get_probs(inter.data.X[:, 0])) <= 1 + 1e-10)
        self.assertTrue(1 - 1e-10 <= np.sum(inter.get_probs(inter.data.X[:, 0], inter.data.Y)) <= 1 + 1e-10)

    def test_probs_sparse_nans(self):
        d = load_artificial_data(3, 10, 2, 2, 10, 3, sparse=10)  # Sparse with nans
        inter = Interactions(d)
        self.assertTrue(1 - 1e-10 <= np.sum(inter.get_probs(inter.data.X[:, 0])) <= 1 + 1e-10)
        self.assertTrue(1 - 1e-10 <= np.sum(inter.get_probs(inter.data.X[:, 0], inter.data.Y)) <= 1 + 1e-10)

    def test_h_empty_list_input(self):
        d = load_artificial_data(3, 10, 2, 2)
        inter = Interactions(d)
        self.assertEqual(inter.h([]), 0)

    def test_h_positive(self):
        d = load_artificial_data(3, 10, 2, 2)  # Random dataset
        inter = Interactions(d)
        self.assertTrue(inter.h(inter.get_probs(inter.data.X[:,0])) >= 0)

    def test_h_zero(self):
        d = load_artificial_data(3, 10, 1, 2)  # Only one unique value in columns.
        inter = Interactions(d)
        self.assertEqual(inter.h(inter.get_probs(inter.data.X[:,0])), 0)

    def test_i_arguments(self):
        d = load_artificial_data(3, 10, 2, 2)  # Random dataset
        inter = Interactions(d)
        self.assertRaises(TypeError, inter.i, inter.data.X[:,0])

    def test_no_info_gain(self):
        d = load_xor_data()
        inter = Interactions(d, alpha=0)
        self.assertEqual(inter.i(inter.data.X[:, 0], inter.data.Y), 0)
        self.assertEqual(inter.i(inter.data.X[:, 1], inter.data.Y), 0)

    def test_max_info_gain(self):
        d = load_xor_data()
        inter = Interactions(d, alpha=0)
        self.assertEqual(inter.i(inter.data.X[:, 0], inter.data.X[:, 1], inter.data.Y), -1)

    def test_attribute_interactions(self):
        d = load_xor_data()
        inter = Interactions(d, alpha=0)
        interaction = inter.attribute_interactions(0,1)
        self.assertEqual(interaction.abs_ig_a, 0)
        self.assertEqual(interaction.abs_ig_b, 0)
        self.assertEqual(interaction.abs_ig_ab, -1)
        self.assertEqual(interaction.abs_total_ig_ab, 1)
        self.assertEqual(interaction.rel_ig_a, 0)
        self.assertEqual(interaction.rel_ig_b, 0)
        self.assertEqual(interaction.rel_ig_ab, -1)
        self.assertEqual(interaction.rel_total_ig_ab, 1)

    def test_attribute_interactions_argument_type(self):
        d = load_artificial_data(3, 10, 2, 2)  # Random dataset
        inter = Interactions(d)
        self.assertRaises(TypeError, inter.attribute_interactions, 2, "string")

    def test_attribute_interactions_argument_range(self):
        d = load_artificial_data(3, 10, 2, 2)  # Random dataset
        inter = Interactions(d)
        self.assertRaises(IndexError, inter.attribute_interactions, 2, 5)

    def test_interaction_matrix(self):
        d = load_xor_data()
        inter = Interactions(d, alpha=0)
        int_M = inter.interaction_matrix().tolist()
        self.assertEqual(int_M, [[0.0, -1.0], [-1.0, 0.0]])

    def test_top_attribute_interactions(self):
        d = Orange.data.Table('zoo')
        inter = Interactions(d)
        int_M = inter.interaction_matrix()
        top3 = inter.get_top_att(3, criteria="total")
        self.assertEqual(top3[0].a_name, "legs")
        self.assertEqual(top3[0].b_name, "milk")
        self.assertEqual(top3[1].a_name, "legs")
        self.assertEqual(top3[1].b_name, "eggs")
        self.assertEqual(top3[2].a_name, "legs")
        self.assertEqual(top3[2].b_name, "hair")

