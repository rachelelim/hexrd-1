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


if __name__ == '__main__':
    unittest.main()
