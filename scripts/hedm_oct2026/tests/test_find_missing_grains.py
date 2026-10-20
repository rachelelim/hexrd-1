"""Tests for the pure parts of nf_hedm/find_missing_grains.py (no data needed)."""
import importlib.util
import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from hexrd import rotations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
SCRIPT = os.path.join(os.path.dirname(HERE), 'nf_hedm', 'find_missing_grains.py')
_spec = importlib.util.spec_from_file_location('find_missing_grains', SCRIPT)
fmg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fmg)

Q_SYM = rotations.quatOfLaueGroup('d6h')
OPS = [R.from_quat([s[1], s[2], s[3], s[0]]) for s in Q_SYM.T]


def rot(angle_deg, axis):
    axis = np.asarray(axis, float)
    return R.from_rotvec(np.radians(angle_deg)*axis/np.linalg.norm(axis))


class TestMergeSimilarOrientations(unittest.TestCase):

    def setUp(self):
        self.a = R.from_rotvec([0.4, 0.1, -0.3])

    def merge(self, rotations_, conf, tol=0.5):
        em = np.array([r.as_rotvec() for r in rotations_])
        return fmg.merge_similar_orientations(em, np.array(conf), tol, Q_SYM)

    def test_close_orientations_merge_and_keep_the_most_confident(self):
        near = rot(0.2, [1, 2, 3]) * self.a
        out = self.merge([self.a, near], [0.6, 0.8])
        self.assertEqual(out.shape, (1, 3))
        np.testing.assert_allclose(out[0], near.as_rotvec())

    def test_distant_orientations_stay_separate(self):
        out = self.merge([self.a, rot(10, [1, 0, 0]) * self.a], [0.6, 0.8])
        self.assertEqual(out.shape, (2, 3))

    def test_crystal_side_symmetry_equivalents_merge(self):
        # a * S is the same crystal orientation as a (same diffraction)
        equivalents = [self.a * op for op in OPS[1:4]]
        out = self.merge([self.a] + equivalents, [0.9, 0.8, 0.7, 0.6])
        self.assertEqual(out.shape, (1, 3))

    def test_sample_side_images_do_not_merge(self):
        # S * a is a different orientation: it scores about 0.1 against the
        # images where a scores 0.9, so it must not be merged with a
        images = [op * self.a for op in OPS[1:4]]
        out = self.merge([self.a] + images, [0.9, 0.8, 0.7, 0.6])
        self.assertEqual(out.shape, (4, 3))

    def test_empty_input(self):
        out = fmg.merge_similar_orientations(np.zeros((0, 3)), np.zeros(0), 0.5, Q_SYM)
        self.assertEqual(out.shape, (0, 3))

    def test_single_orientation(self):
        out = self.merge([self.a], [0.8])
        self.assertEqual(out.shape, (1, 3))

    def test_without_symmetry_operators(self):
        em = np.array([self.a.as_rotvec(), (self.a * OPS[1]).as_rotvec()])
        out = fmg.merge_similar_orientations(em, np.array([0.9, 0.8]), 0.5, None)
        self.assertEqual(out.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
