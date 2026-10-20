"""Tests for refining along the marching (nfutil.spatial_adaptive.march_refined), no data needed.

A Voronoi microstructure whose grains have an orientation gradient inside them, and a
fake confidence that peaks at the true local orientation, so the marching can be
checked against the truth: every voxel is reached, the orientation follows the
gradient, and far fewer orientations are scored than refining every voxel alone.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfutil.processor_functions import _pattern_search_refine  # noqa: E402
from nfutil.spatial_adaptive import march_refined  # noqa: E402

GLOBAL_RADII = (1.0, 0.5, 0.25, 0.1)
LOCAL_RADII = (0.25, 0.1)
WIDTH = 0.01    # the confidence is zero this far (rad) from the true orientation
GRADIENT = 0.00015  # change of orientation per voxel inside a grain (rad): about 0.2 degrees across a grain, as on real data


class Phantom:

    def __init__(self, shape=(1, 50, 60), n_grains=6, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        sites = np.column_stack([rng.uniform(0, shape[1], n_grains), rng.uniform(0, shape[2], n_grains)])
        d = (yy[..., None] - sites[:, 0])**2 + (xx[..., None] - sites[:, 1])**2
        self.grain = d.argmin(-1)[None]
        self.mask = ((yy - shape[1]/2)**2 + (xx - shape[2]/2)**2 < (min(shape[1:])/2 - 1)**2)[None]
        base = rng.normal(scale=0.5, size=(n_grains, 3))
        direction = rng.normal(size=(n_grains, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        # the true orientation at every voxel: the grain's, plus a gradient across it
        along = (xx + 0.5*yy)[None] - (xx + 0.5*yy).mean()
        self.truth = (base[self.grain] + direction[self.grain]*GRADIENT*along[..., None])[self.mask]
        self.grain_flat = self.grain[self.mask]
        # what an FF list would hold: each grain's orientation, a little off
        self.ff = base + rng.normal(scale=0.001, size=base.shape)
        self.n_scored = 0

    def confidence(self, voxels, candidates):
        """Confidence of each candidate orientation at each voxel."""
        self.n_scored += candidates.shape[0]
        return np.clip(1. - np.linalg.norm(candidates - self.truth[voxels], axis=1)/WIDTH, 0., 1.)

    def global_fn(self, voxels):
        em = np.zeros((len(voxels), 3)); conf = np.zeros(len(voxels)); origin = np.zeros(len(voxels), int)
        n_tests = 0
        for i, v in enumerate(voxels):
            scores = np.array([self.confidence(v, self.ff[k][None])[0] for k in range(len(self.ff))])
            n_tests += len(self.ff)
            origin[i] = int(np.argmax(scores))
            em[i], conf[i] = _pattern_search_refine(
                self.ff[origin[i]], lambda t, v=v: self.confidence(v, t), GLOBAL_RADII)
        return em, conf, origin, n_tests, 0

    def make_local_fn(self, drop_off=0.8):
        def local_fn(voxels, candidates, reference):
            n = len(voxels)
            em = np.zeros((n, 3)); conf = np.zeros(n); pick = np.zeros(n, int); accepted = np.zeros(n, bool)
            for i, v in enumerate(voxels):
                options = np.atleast_2d(candidates[i])
                scores = np.array([self.confidence(v, o[None])[0] for o in options])
                pick[i] = int(np.argmax(scores))
                em[i], conf[i] = options[pick[i]], scores[pick[i]]
                if scores[pick[i]] >= drop_off*reference[i]:
                    accepted[i] = True
                    em[i], conf[i] = _pattern_search_refine(
                        options[pick[i]], lambda t, v=v: self.confidence(v, t), LOCAL_RADII,
                        start_confidence=scores[pick[i]])
            return em, conf, pick, accepted, 0
        return local_fn


class TestMarchRefined(unittest.TestCase):

    def setUp(self):
        self.p = Phantom()
        self.em, self.conf, self.origin, self.stats = march_refined(
            self.p.mask, self.p.global_fn, self.p.make_local_fn(), seed_spacing=8)
        self.error = np.linalg.norm(self.em - self.p.truth, axis=1)

    def test_every_voxel_gets_an_orientation(self):
        self.assertEqual(len(self.em), int(self.p.mask.sum()))
        self.assertTrue((self.origin >= 0).all())
        self.assertTrue((self.conf > 0).all())

    def test_orientations_are_close_to_the_truth(self):
        # the finest search radius is 0.1 degrees = 0.0017 rad
        self.assertLess(np.percentile(self.error, 99), 0.004)

    def test_follows_the_gradient_inside_a_grain(self):
        # Each grain's own orientation is wrong by up to this much at its far end
        ff_error = np.linalg.norm(self.p.ff[self.p.grain_flat] - self.p.truth, axis=1)
        self.assertGreater(ff_error.max(), 0.004)
        self.assertGreater(ff_error.max(), 1.5*self.error.max())
        self.assertLess(self.error.mean(), 0.5*ff_error.mean())

    def test_each_voxel_comes_from_its_own_grain(self):
        self.assertGreater((self.origin == self.p.grain_flat).mean(), 0.97)

    def test_neighbors_do_most_of_the_work(self):
        self.assertGreater(self.stats['n_local'], 0.5*self.stats['n_voxels'])
        self.assertEqual(self.stats['n_seed'] + self.stats['n_global'] + self.stats['n_local'],
                         self.stats['n_voxels'])

    def test_far_fewer_orientations_scored_than_refining_every_voxel(self):
        scored = self.p.n_scored
        alone = Phantom()
        for v in range(int(alone.mask.sum())):
            _pattern_search_refine(alone.ff[alone.grain_flat[v]], lambda t, v=v: alone.confidence(v, t),
                                   GLOBAL_RADII)
        self.assertLess(scored, 0.5*alone.n_scored)

    def test_island_cut_off_from_the_seeds_is_reached(self):
        mask = np.zeros((1, 20, 20), bool)
        mask[0, :5, :5] = True
        mask[0, 15:, 15:] = True
        p = Phantom(shape=(1, 20, 20), n_grains=2, seed=3)
        p.mask = mask
        p.truth = p.truth[:0]    # rebuilt below for the smaller mask
        rng = np.random.default_rng(3)
        base = rng.normal(scale=0.5, size=(2, 3))
        grain = np.zeros(mask.shape, int); grain[0, 15:, 15:] = 1
        p.truth = base[grain[mask]]
        p.ff = base + 0.0005
        em, conf, origin, stats = march_refined(mask, p.global_fn, p.make_local_fn(), seed_spacing=100)
        self.assertTrue((conf > 0).all())
        np.testing.assert_array_equal(origin, grain[mask])


if __name__ == '__main__':
    unittest.main()
