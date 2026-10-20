"""Tests for nfutil.spatial_adaptive (no experimental data needed)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfutil.spatial_adaptive import march, neighbor_table  # noqa: E402


def voronoi_phantom(shape=(1, 60, 70), n_grains=8, seed=0):
    """Grain id of every voxel of a Voronoi microstructure, and a disc mask."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]),
                         indexing='ij')
    sites = np.column_stack([rng.uniform(0, shape[1], n_grains),
                             rng.uniform(0, shape[2], n_grains)])
    dist = ((yy[..., None] - sites[:, 0])**2
            + (xx[..., None] - sites[:, 1])**2)
    truth = dist.argmin(-1)[None]
    mask = ((yy - shape[1]/2)**2 + (xx - shape[2]/2)**2
            < (min(shape[1:])/2 - 1)**2)[None]
    return truth, mask


def make_test_fn(truth_flat, boundary_flat, calls):
    """A confidence that is high for the true grain (lower on boundaries)."""
    def test_fn(voxels, bags):
        best = np.zeros(len(voxels), int)
        conf = np.zeros(len(voxels))
        n_tests = 0
        for i, (v, bag) in enumerate(zip(voxels, bags)):
            true_conf = 0.55 if boundary_flat[v] else 0.9
            c = np.where(bag == truth_flat[v], true_conf, 0.1)
            best[i] = bag[np.argmax(c)]
            conf[i] = c.max()
            n_tests += len(bag)
        calls.append(n_tests)
        return best, conf, n_tests
    return test_fn


class TestNeighborTable(unittest.TestCase):

    def test_six_connectivity(self):
        mask = np.ones((1, 3, 3), bool)
        nbrs = neighbor_table(mask, 6)
        self.assertEqual(nbrs.shape, (9, 6))
        centre = nbrs[4]            # voxel (0, 1, 1)
        self.assertEqual(sorted(centre[centre >= 0]), [1, 3, 5, 7])
        corner = nbrs[0]
        self.assertEqual(sorted(corner[corner >= 0]), [1, 3])

    def test_mask_edges_are_not_neighbors(self):
        mask = np.zeros((1, 3, 3), bool)
        mask[0, 1, :] = True
        nbrs = neighbor_table(mask, 6)
        self.assertEqual(nbrs.shape[0], 3)
        self.assertEqual(sorted(nbrs[1][nbrs[1] >= 0]), [0, 2])


class TestMarch(unittest.TestCase):

    def setUp(self):
        truth, self.mask = voronoi_phantom()
        self.truth_flat = truth[self.mask]
        # Voxels with a neighbor in another grain have weaker signal
        nbrs = neighbor_table(self.mask)
        other = self.truth_flat[np.where(nbrs >= 0, nbrs, 0)] \
            != self.truth_flat[:, None]
        self.boundary = (other & (nbrs >= 0)).any(1)

    def test_recovers_every_voxel(self):
        calls = []
        fn = make_test_fn(self.truth_flat, self.boundary, calls)
        idx, conf, stats = march(self.mask, 8, fn)
        np.testing.assert_array_equal(idx, self.truth_flat)
        self.assertTrue((conf > 0).all())

    def test_far_fewer_tests_than_all_grains(self):
        calls = []
        fn = make_test_fn(self.truth_flat, self.boundary, calls)
        _, _, stats = march(self.mask, 8, fn)
        self.assertEqual(stats['n_tests'], sum(calls))
        self.assertLess(stats['n_tests'], 0.4*stats['n_voxels']*8)

    def test_every_voxel_assigned_when_nothing_converges(self):
        # Confidence below the converged threshold everywhere: nothing can
        # spread, so every voxel falls back to testing every candidate.
        calls = []
        fn = make_test_fn(self.truth_flat, np.ones_like(self.boundary), calls)
        idx, _, stats = march(self.mask, 8, fn, converged_confidence=0.7)
        np.testing.assert_array_equal(idx, self.truth_flat)
        self.assertEqual(stats['n_tests'], stats['n_voxels']*8)

    def test_isolated_island_is_reached(self):
        mask = np.zeros((1, 20, 20), bool)
        mask[0, :5, :5] = True
        mask[0, 15:, 15:] = True        # not connected to the first block
        truth = np.zeros(mask.shape, int)
        truth[0, 15:, 15:] = 1
        truth_flat = truth[mask]
        fn = make_test_fn(truth_flat, np.zeros(len(truth_flat), bool), [])
        idx, _, _ = march(mask, 2, fn, seed_spacing=100)
        np.testing.assert_array_equal(idx, truth_flat)


if __name__ == '__main__':
    unittest.main()
