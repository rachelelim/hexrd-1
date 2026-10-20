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
from nfutil.spatial_adaptive import boundary_band, march_refined, neighbor_table, refine_boundaries  # noqa: E402

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


class TestBoundaryPass(unittest.TestCase):

    def band_fn(self, phantom):
        """The phantom's global_fn, testing only the grains in each voxel's bag."""
        def fn(voxels, bags):
            em = np.zeros((len(voxels), 3)); conf = np.zeros(len(voxels)); origin = np.zeros(len(voxels), int)
            for i, (v, bag) in enumerate(zip(voxels, bags)):
                scores = np.array([phantom.confidence(v, phantom.ff[k][None])[0] for k in bag])
                origin[i] = bag[int(np.argmax(scores))]
                em[i], conf[i] = _pattern_search_refine(
                    phantom.ff[origin[i]], lambda t, v=v: phantom.confidence(v, t), GLOBAL_RADII)
            return em, conf, origin, 0, 0
        return fn

    def test_band_of_two_grains(self):
        mask = np.ones((1, 4, 8), bool)
        origin = np.where(np.arange(8) < 4, 3, 7)[None, :].repeat(4, axis=0).ravel()
        nbrs = neighbor_table(mask)
        for width, columns in ((1, [3, 4]), (2, [2, 3, 4, 5]), (3, [1, 2, 3, 4, 5, 6])):
            voxels, bags = boundary_band(origin, nbrs, width)
            self.assertEqual(sorted(set(voxels % 8)), columns)
            self.assertEqual(len(voxels), 4*len(columns))
            for bag in bags:
                np.testing.assert_array_equal(bag, [3, 7])   # both grains, from the voxels touching the boundary and from those behind

    def test_no_boundary_in_one_grain(self):
        mask = np.ones((1, 5, 5), bool)
        voxels, bags = boundary_band(np.zeros(25, int), neighbor_table(mask), 2)
        self.assertEqual(len(voxels), 0)

    def test_restores_boundary_voxels_the_march_gave_to_the_wrong_grain(self):
        p = Phantom()
        em, conf, origin, _ = march_refined(p.mask, p.global_fn, p.make_local_fn(), seed_spacing=8)
        nbrs = neighbor_table(p.mask)
        voxels, _ = boundary_band(origin, nbrs, 1)
        # let the neighboring grain spread one voxel into every boundary voxel it can, as the march does
        wrong = []
        for v in voxels:
            others = [origin[u] for u in nbrs[v] if u >= 0 and origin[u] != origin[v]]
            if others and origin[v] == p.grain_flat[v]:
                origin[v] = others[0]
                em[v] = p.ff[others[0]]
                conf[v] = 0.3
                wrong.append(v)
        self.assertGreater(len(wrong), 20)
        outside = np.setdiff1d(np.arange(len(origin)), boundary_band(origin, nbrs, 2)[0])
        before = origin.copy(), em.copy()
        stats = refine_boundaries(em, conf, origin, nbrs, self.band_fn(p), width=2)
        self.assertGreater((origin[wrong] == p.grain_flat[wrong]).mean(), 0.9)
        self.assertGreater((origin == p.grain_flat).mean(), 0.97)
        self.assertGreater(stats['n_band_changed'], 0.9*len(wrong))
        self.assertEqual(stats['n_band'], len(boundary_band(before[0], nbrs, 2)[0]))
        # voxels away from the boundary are not touched
        np.testing.assert_array_equal(origin[outside], before[0][outside])
        np.testing.assert_array_equal(em[outside], before[1][outside])


if __name__ == '__main__':
    unittest.main()
