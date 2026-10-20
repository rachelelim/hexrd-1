"""Tests for the coarse-to-fine orientation refinement."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfutil.processor_functions import _pattern_search_refine  # noqa: E402

RADII = [5.0, 4., 3.5, 3., 2.5, 2., 1.5, 1.0, 0.5, 0.25, 0.1]


def bowl(target):
    """Confidence that peaks at ``target`` (exp-map) and falls off smoothly."""
    def cost(trial):
        return 1.0 - np.linalg.norm(trial - target, axis=1)
    return cost


class TestPatternSearchRefine(unittest.TestCase):

    def test_recentering_walks_past_one_radius(self):
        # 40 degrees away is further than a single pass per radius can reach
        # (the radii sum to ~23 degrees along one axis).
        start = np.zeros(3)
        target = np.array([np.radians(40.), 0., 0.])
        exp_map, conf = _pattern_search_refine(start, bowl(target), RADII)
        self.assertLess(np.degrees(np.linalg.norm(exp_map - target)), 0.1)

        # With re-centering disabled it stalls well short of the target
        exp_map1, _ = _pattern_search_refine(
            start, bowl(target), RADII, max_recenters=1)
        self.assertGreater(np.degrees(np.linalg.norm(exp_map1 - target)), 10.)

    def test_confidence_never_decreases(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            start = rng.normal(scale=0.1, size=3)
            target = start + rng.normal(scale=0.05, size=3)
            cost = bowl(target)
            exp_map, conf = _pattern_search_refine(start, cost, RADII)
            self.assertGreaterEqual(conf, cost(start[None])[0])
            self.assertAlmostEqual(conf, cost(exp_map[None])[0])

    def test_terminates_when_nothing_improves(self):
        calls = []

        def flat(trial):
            calls.append(len(trial))
            return np.ones(len(trial))

        _pattern_search_refine(np.zeros(3), flat, RADII)
        # One pass to take, one to find no improvement, per radius
        self.assertEqual(len(calls), 2 * len(RADII))


def old_pattern_search_refine(start_exp_map, cost, radii_deg, max_recenters=100):
    """The pattern search as it was before points stopped being scored twice."""
    exp_map = np.asarray(start_exp_map, dtype=float)
    confidence = 0.0
    for mis in radii_deg:
        mis_amt = np.radians(mis)
        ori_pts = np.array([-mis_amt, 0., mis_amt])
        XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts)
        grid0 = np.vstack([XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T
        for counter in range(max_recenters):
            all_confidence = cost(grid0 + exp_map)
            idx = np.argmax(all_confidence)
            if counter > 0 and all_confidence[idx] <= confidence:
                break
            exp_map = grid0[idx] + exp_map
            confidence = all_confidence[idx]
    return exp_map, confidence


class TestPatternSearchSavesWork(unittest.TestCase):
    """Same answer as before, with fewer orientations scored."""

    def costs(self):
        rng = np.random.default_rng(0)
        for _ in range(8):
            target = rng.normal(scale=0.05, size=3)
            yield bowl(target)
        # Flat stretches and ties, where which point wins depends on the order
        yield lambda t: np.round(1.0 - np.linalg.norm(t - 0.01, axis=1), 2)
        yield lambda t: np.ones(len(t))
        # Noisy, but a fixed value at each point
        def noisy(t):
            return np.array([0.5 + 0.1*np.sin(1e4*x.sum()) - np.linalg.norm(x) for x in t])
        yield noisy

    def test_same_result_as_scoring_every_point_every_time(self):
        rng = np.random.default_rng(1)
        for cost in self.costs():
            start = rng.normal(scale=0.1, size=3)
            new = _pattern_search_refine(start, cost, RADII)
            old = old_pattern_search_refine(start, cost, RADII)
            np.testing.assert_array_equal(new[0], old[0])
            self.assertEqual(new[1], old[1])

    def test_fewer_orientations_are_scored(self):
        # Refining an FF orientation starts about 0.1 degrees from where it ends up
        rng = np.random.default_rng(2)
        start = np.array([0.02, -0.01, 0.03])
        ratios = []
        for _ in range(10):
            cost = bowl(start + rng.normal(scale=0.002, size=3))
            scored = {'new': 0, 'old': 0}

            def counting(key):
                def f(trial):
                    scored[key] += len(trial)
                    return cost(trial)
                return f
            _pattern_search_refine(start, counting('new'), RADII)
            old_pattern_search_refine(start, counting('old'), RADII)
            self.assertLess(scored['new'], scored['old'])
            ratios.append(scored['new']/scored['old'])
        # About half: the second pass at each radius is skipped when nothing moved
        self.assertLess(np.mean(ratios), 0.6)

    def test_never_scores_more_than_before(self):
        # Including searches that keep moving, ties and flat costs
        start = np.array([0.02, -0.01, 0.03])
        for cost in self.costs():
            scored = {'new': 0, 'old': 0}

            def counting(key):
                def f(trial):
                    scored[key] += len(trial)
                    return cost(trial)
                return f
            _pattern_search_refine(start, counting('new'), RADII)
            old_pattern_search_refine(start, counting('old'), RADII)
            self.assertLessEqual(scored['new'], scored['old'])

    def test_no_point_is_scored_twice_in_a_pass(self):
        seen = []

        def cost(trial):
            seen.extend(map(tuple, trial))
            return 1.0 - np.linalg.norm(trial - 0.03, axis=1)
        _pattern_search_refine(np.zeros(3), cost, RADII)
        # re-centering overlaps neighbouring grids, so some points repeat, but
        # the center of a grid (the current best) is never scored again at its
        # own radius
        self.assertLess(len(seen), 27*len(RADII)*2)


if __name__ == '__main__':
    unittest.main()
