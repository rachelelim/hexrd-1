"""Tests for the pure helpers in nfutil.orientation_search (no data needed)."""
import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from hexrd import rotations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfutil.orientation_search import (  # noqa: E402
    merge_orientations, misorientation_deg, separate_candidates)

Q_SYM = rotations.quatOfLaueGroup('d6h')


def rotate(exp_map, degrees, axis):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    return (R.from_rotvec(np.radians(degrees)*axis)
            * R.from_rotvec(exp_map)).as_rotvec()


class TestMisorientation(unittest.TestCase):

    def test_known_angle(self):
        a = np.array([0.3, -0.2, 0.5])
        b = rotate(a, 1.5, [1, 2, 3])
        self.assertAlmostEqual(misorientation_deg(a, b[None], Q_SYM)[0], 1.5, places=4)

    def test_zero_for_same_orientation(self):
        a = np.array([0.3, -0.2, 0.5])
        self.assertAlmostEqual(misorientation_deg(a, a[None], Q_SYM)[0], 0.0, places=4)

    def test_symmetry_equivalent_is_zero(self):
        # A 60 degree turn about the hexagonal c axis of the crystal (multiplying
        # on the crystal side) is a symmetry operation
        a = np.array([0.3, -0.2, 0.5])
        b = (R.from_rotvec(a) * R.from_rotvec(np.radians(60)*np.array([0., 0., 1.]))).as_rotvec()
        self.assertGreater(np.linalg.norm(a - b), 0.5)
        self.assertAlmostEqual(misorientation_deg(a, b[None], Q_SYM)[0], 0.0, places=3)

    def test_many(self):
        a = np.zeros(3) + 0.1
        many = np.array([rotate(a, d, [0, 0, 1]) for d in (0.5, 1.0, 2.0)])
        np.testing.assert_allclose(misorientation_deg(a, many, Q_SYM), [0.5, 1.0, 2.0], atol=1e-3)


class TestSeparateCandidates(unittest.TestCase):

    def setUp(self):
        self.a = np.array([0.4, 0.1, -0.3])
        self.b = rotate(self.a, 10.0, [1, 0, 0])
        rng = np.random.default_rng(0)
        # Five points within 0.3 degrees of a, three within 0.3 of b
        near_a = [rotate(self.a, 0.3*rng.random(), rng.normal(size=3)) for _ in range(5)]
        near_b = [rotate(self.b, 0.3*rng.random(), rng.normal(size=3)) for _ in range(3)]
        self.exp_maps = np.array(near_a + near_b)
        self.conf = np.array([0.5, 0.9, 0.6, 0.55, 0.52, 0.3, 0.8, 0.35])

    def test_one_per_cluster_best_first(self):
        kept = separate_candidates(self.exp_maps, self.conf, Q_SYM, 1.0, 10)
        self.assertEqual(list(kept), [1, 6])

    def test_limit(self):
        kept = separate_candidates(self.exp_maps, self.conf, Q_SYM, 1.0, 1)
        self.assertEqual(list(kept), [1])

    def test_small_separation_keeps_more(self):
        kept = separate_candidates(self.exp_maps, self.conf, Q_SYM, 1e-4, 10)
        self.assertEqual(len(kept), len(self.exp_maps))

    def test_symmetry_equivalents_are_one_cluster(self):
        flipped = (R.from_rotvec(self.a)
                   * R.from_rotvec(np.radians(60)*np.array([0., 0., 1.]))).as_rotvec()
        em = np.array([self.a, flipped])
        kept = separate_candidates(em, np.array([0.5, 0.7]), Q_SYM, 1.0, 5)
        self.assertEqual(list(kept), [1])


class TestMerge(unittest.TestCase):

    def test_merge_keeps_best_of_each_group(self):
        a = np.array([0.4, 0.1, -0.3])
        em = np.array([a, rotate(a, 0.4, [0, 1, 0]), rotate(a, 20, [1, 1, 0])])
        kept = merge_orientations(em, np.array([0.6, 0.8, 0.7]), Q_SYM, 1.0)
        self.assertEqual(sorted(kept), [1, 2])

    def test_empty(self):
        self.assertEqual(len(merge_orientations(np.zeros((0, 3)), np.zeros(0), Q_SYM, 1.0)), 0)


if __name__ == '__main__':
    unittest.main()
