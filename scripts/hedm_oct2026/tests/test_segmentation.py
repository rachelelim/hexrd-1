"""Tests for nfutil.segmentation (no data needed)."""
import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from hexrd import rotations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfutil.orientation_search import misorientation_deg  # noqa: E402
from nfutil.segmentation import (  # noqa: E402
    grain_average_orientations, misorientation_from_average, misorientation_pairs,
    neighbor_misorientations, quaternions_of_exp_maps, segment_orientation_field)

Q_SYM = rotations.quatOfLaueGroup('d6h')
OPS = [R.from_quat([s[1], s[2], s[3], s[0]]) for s in Q_SYM.T]


def exp_map(rotation):
    return rotation.as_rotvec()


def turn(degrees, axis):
    axis = np.asarray(axis, float)
    return R.from_rotvec(np.radians(degrees)*axis/np.linalg.norm(axis))


class TestMisorientationPairs(unittest.TestCase):

    def test_matches_the_one_at_a_time_function(self):
        rng = np.random.default_rng(0)
        a = R.random(60, random_state=1)
        b = R.random(60, random_state=2)
        got = misorientation_pairs(quaternions_of_exp_maps(a.as_rotvec()), quaternions_of_exp_maps(b.as_rotvec()), Q_SYM)
        want = [misorientation_deg(x.as_rotvec(), y.as_rotvec()[None], Q_SYM)[0] for x, y in zip(a, b)]
        np.testing.assert_allclose(got, want, atol=1e-6)

    def test_crystal_side_equivalents_are_zero_and_sample_side_images_are_not(self):
        a = R.random(1, random_state=4)[0]
        qa = quaternions_of_exp_maps(exp_map(a))
        for op in OPS[1:6]:
            same = misorientation_pairs(qa, quaternions_of_exp_maps(exp_map(a*op)), Q_SYM)[0]
            other = misorientation_pairs(qa, quaternions_of_exp_maps(exp_map(op*a)), Q_SYM)[0]
            self.assertAlmostEqual(same, 0., places=3)
            self.assertGreater(other, 10.)

    def test_known_angle(self):
        a = R.from_rotvec([0.3, -0.1, 0.2])
        b = turn(1.25, [1, 2, 3])*a
        got = misorientation_pairs(quaternions_of_exp_maps(exp_map(a)), quaternions_of_exp_maps(exp_map(b)), Q_SYM)
        self.assertAlmostEqual(got[0], 1.25, places=4)


def two_grains(shape=(1, 20, 30)):
    """Left half one grain, right half another, each with a small gradient, and each voxel
    written as a random symmetry-equivalent of its orientation."""
    mask = np.ones(shape, bool)
    xx = np.indices(shape)[2][mask]
    rng = np.random.default_rng(1)
    base = [R.from_rotvec([0.4, 0.1, -0.3]), R.from_rotvec([-0.5, 0.6, 0.2])]
    grain = (xx >= shape[2]//2).astype(int)
    rotations_ = [turn(0.01*x, [1, 1, 0])*base[g] for x, g in zip(xx, grain)]       # 0.01 degrees per voxel
    rotations_ = [r*OPS[rng.integers(len(OPS))] for r in rotations_]
    return mask, grain, np.array([exp_map(r) for r in rotations_])


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.mask, self.grain, self.exp_maps = two_grains()

    def test_two_grains_found_even_though_orientations_are_written_as_equivalents(self):
        labels = segment_orientation_field(self.exp_maps, self.mask, Q_SYM, threshold_deg=1.0)
        self.assertEqual(labels.max() + 1, 2)
        # each region is one of the grains
        for g in (0, 1):
            self.assertEqual(len(np.unique(labels[self.grain == g])), 1)

    def test_numbered_by_decreasing_size(self):
        mask = np.ones((1, 10, 20), bool)
        em = np.zeros((200, 3)) + 0.2
        em[:20] = [0.9, -0.3, 0.1]        # a small grain at one end
        labels = segment_orientation_field(em, mask, Q_SYM, threshold_deg=1.0)
        sizes = np.bincount(labels)
        self.assertEqual(list(sizes), sorted(sizes, reverse=True))

    def test_a_threshold_below_the_gradient_splits_a_grain(self):
        # 0.01 degrees per voxel: a threshold under that cuts every step
        labels = segment_orientation_field(self.exp_maps, self.mask, Q_SYM, threshold_deg=0.005)
        self.assertGreater(labels.max() + 1, 2)

    def test_a_threshold_above_the_boundary_merges_the_grains(self):
        boundary = misorientation_deg(self.exp_maps[0], self.exp_maps[-1][None], Q_SYM)[0]
        labels = segment_orientation_field(self.exp_maps, self.mask, Q_SYM, threshold_deg=boundary + 5.)
        self.assertEqual(labels.max() + 1, 1)

    def test_voxels_not_touching_are_not_joined(self):
        mask = np.zeros((1, 10, 10), bool)
        mask[0, :3, :3] = True
        mask[0, 7:, 7:] = True
        em = np.zeros((int(mask.sum()), 3)) + 0.3     # the same orientation everywhere
        labels = segment_orientation_field(em, mask, Q_SYM, threshold_deg=1.0)
        self.assertEqual(labels.max() + 1, 2)

    def test_neighbor_misorientations_count_and_size(self):
        d = neighbor_misorientations(self.exp_maps, self.mask, Q_SYM)
        shape = self.mask.shape
        self.assertEqual(len(d), shape[1]*(shape[2] - 1) + (shape[1] - 1)*shape[2])
        # steps inside a grain are the 0.01 degree gradient, and one column of pairs is the boundary
        self.assertLess(np.median(d), 0.05)
        self.assertEqual(int((d > 5).sum()), shape[1])


class TestGrainAverage(unittest.TestCase):

    def test_average_and_distance_from_it(self):
        rng = np.random.default_rng(3)
        truth = R.from_rotvec([0.4, 0.1, -0.3])
        angles = rng.uniform(0.02, 0.3, 80)
        axes = rng.normal(size=(80, 3))
        members = [turn(a, ax)*truth for a, ax in zip(angles, axes)]
        # written as random equivalents
        written = [m*OPS[rng.integers(len(OPS))] for m in members]
        em = np.array([exp_map(w) for w in written])
        labels = np.zeros(80, int)
        average = grain_average_orientations(em, labels, Q_SYM)
        # the average is near the truth: the spread is a fraction of a degree in random directions
        d = misorientation_deg(exp_map(truth), average, Q_SYM)[0]
        self.assertLess(d, 0.1)
        # each voxel's distance from the average is about the angle it was turned by
        grod = misorientation_from_average(em, labels, average, Q_SYM)
        np.testing.assert_allclose(grod, angles, atol=0.1)

    def test_every_region_is_averaged_on_its_own(self):
        a, b = R.from_rotvec([0.4, 0.1, -0.3]), R.from_rotvec([-0.5, 0.6, 0.2])
        em = np.array([exp_map(a)]*5 + [exp_map(b)]*5)
        labels = np.array([0]*5 + [1]*5)
        averages = grain_average_orientations(em, labels, Q_SYM)
        self.assertEqual(averages.shape, (2, 3))
        grod = misorientation_from_average(em, labels, averages, Q_SYM)
        np.testing.assert_allclose(grod, 0., atol=1e-4)

    def test_weights_pull_the_average(self):
        a = R.from_rotvec([0.4, 0.1, -0.3])
        em = np.array([exp_map(a), exp_map(turn(1.0, [0, 0, 1])*a)])
        labels = np.zeros(2, int)
        heavy_first = grain_average_orientations(em, labels, Q_SYM, weights=np.array([10., 1.]))
        heavy_second = grain_average_orientations(em, labels, Q_SYM, weights=np.array([1., 10.]))
        to_first = [misorientation_deg(em[0], x[None], Q_SYM)[0] for x in (heavy_first[0], heavy_second[0])]
        self.assertLess(to_first[0], to_first[1])
        # an unweighted average is halfway
        even = grain_average_orientations(em, labels, Q_SYM)
        self.assertAlmostEqual(misorientation_deg(em[0], even, Q_SYM)[0], 0.5, places=2)


if __name__ == '__main__':
    unittest.main()
