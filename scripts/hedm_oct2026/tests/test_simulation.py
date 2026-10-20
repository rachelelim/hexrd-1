"""nfutil.simulation.simulate_angles must return exactly what simulateGVecs does.

The refinement and the search depend on it, and the speedup is only worth having if
no score changes. Needs a material, so it reads the Ti-7Al config (and, if present,
the cubic lowO config) from the NF test data, and is skipped without them.
"""
import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TI7AL_DIR = os.environ.get(
    'NF_TEST_DATA_DIR',
    '/Users/lim37/Library/CloudStorage/OneDrive-LLNL/ti7al_cyclic_HEDM/ti7-1/NF')
TI7AL_CONFIG = os.path.join(HERE, 'data', 'ti7al_S0_L0_masked.yml')
LOWO_DIR = os.environ.get(
    'NF_TEST_CUBIC_DATA_DIR',
    '/Users/lim37/Library/CloudStorage/OneDrive-LLNL/Vanadium/CHESS-2026-03/lowO-comp-1')
LOWO_CONFIG = 'nf_config_S0_L2.yml'

UNSTRAINED = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])
PANEL = [(-10, -10), (10, 10)]


def reference_angles(experiment, exp_map):
    from hexrd import xrdutil
    return xrdutil.simulateGVecs(
        experiment.plane_data, experiment.detector_params, np.hstack([exp_map, UNSTRAINED]),
        panel_dims=PANEL, pixel_pitch=experiment.pixel_size, ome_range=experiment.ome_range,
        ome_period=experiment.ome_period, distortion=None)[2]


class SimulationCases:
    folder = None
    config = None

    @classmethod
    def setUpClass(cls):
        from hexrd.hedm.grainmap import nfutil
        from hexrd.hedm import nf_config
        cls._cwd = os.getcwd()
        os.chdir(cls.folder)
        cls.addClassCleanup(os.chdir, cls._cwd)
        cls.experiment = nfutil.Experiment(nf_config.open_file(cls.config)[0])
        cls.orientations = R.random(120, random_state=3).as_rotvec()

    def check(self, experiment):
        from hexrd.hedm.grainmap.nfutil.simulation import simulate_angles
        for exp_map in self.orientations:
            expected = reference_angles(experiment, exp_map)
            got = simulate_angles(experiment, exp_map)
            self.assertEqual(got.shape, expected.shape)
            # Some rows are NaN (reflections with no solution) in both
            np.testing.assert_array_equal(got, expected)

    def test_all_rings(self):
        self.check(self.experiment)

    def test_ring_subsets(self):
        from hexrd.hedm.grainmap.nfutil.orientation_search import ring_subset_experiment
        for n_rings in (3, 8):
            self.check(ring_subset_experiment(self.experiment, n_rings))

    def test_precompute_uses_the_same_peaks(self):
        # the precompute functions feed everything downstream
        from hexrd.hedm.grainmap import nfutil
        from hexrd.hedm.grainmap.nfutil.processor_functions import (
            _precompute_diffraction_data_of_single_orientation as one,
            _precompute_diffraction_data_of_many_orientations as many)
        subset = self.orientations[:5]
        many_data = many(self.experiment, subset)
        for i, exp_map in enumerate(subset):
            np.testing.assert_array_equal(one(self.experiment, exp_map)[1], many_data[1][i])
            np.testing.assert_array_equal(many_data[1][i], reference_angles(self.experiment, exp_map))


@unittest.skipUnless(os.path.isfile(os.path.join(TI7AL_DIR, 'retiga.yml')), 'Ti-7Al test data not found')
class TestTi7Al(SimulationCases, unittest.TestCase):
    folder = TI7AL_DIR
    config = TI7AL_CONFIG


@unittest.skipUnless(os.path.isfile(os.path.join(LOWO_DIR, LOWO_CONFIG)), 'lowO test data not found')
class TestLowO(SimulationCases, unittest.TestCase):
    folder = LOWO_DIR
    config = LOWO_CONFIG


if __name__ == '__main__':
    unittest.main()
