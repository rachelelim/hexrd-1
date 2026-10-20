"""Regression test: single-layer NF reconstruction of lowO-comp-1.

Runs the same steps as nf_hedm/reconstruct_nf.py in-process (without saving)
and compares the grain and confidence maps against a stored reference.

The data folder is read from NF_TEST_DATA_DIR (default below). The test is
skipped if the folder is not available. Run with either:

    python -m unittest discover -s tests
    pytest tests
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

DATA_DIR = os.environ.get(
    'NF_TEST_DATA_DIR',
    '/Users/lim37/Library/CloudStorage/OneDrive-LLNL/Vanadium/CHESS-2026-03/'
    'lowO-comp-1')
CONFIG = 'nf_config_S0_L2.yml'
REFERENCE = os.path.join(HERE, 'data', 'lowO_comp_1_S0_L2_reference.npz')


@unittest.skipUnless(os.path.isfile(os.path.join(DATA_DIR, CONFIG)),
                     f'NF test data not found in {DATA_DIR}')
class TestSingleLayerReconstruction(unittest.TestCase):

    def setUp(self):
        # The config uses paths relative to the data folder
        self._cwd = os.getcwd()
        os.chdir(DATA_DIR)
        self.addCleanup(os.chdir, self._cwd)

    def _reconstruct(self, experiment, image_stack, configuration):
        import nfutil
        controller = nfutil.build_controller(configuration)
        Xs, Ys, Zs, mask, test_coordinates = \
            experiment.generate_test_coordinates()
        precomputed = nfutil.precompute_diffraction_data(
            experiment, controller, experiment.exp_maps)
        _, raw_confidence, raw_idx = nfutil.test_orientations_at_coordinates(
            experiment, controller, image_stack, precomputed,
            test_coordinates, refine_yes_no=0)
        return nfutil.process_raw_data(
            raw_confidence, raw_idx, Xs.shape, mask=mask,
            id_remap=experiment.remap)

    def test_matches_reference(self):
        import nfutil
        from hexrd.hedm import nf_config

        configuration = nf_config.open_file(CONFIG)[0]
        experiment = nfutil.Experiment(configuration)
        grain_map, confidence_map = self._reconstruct(
            experiment, experiment.image_stack, configuration)

        ref = np.load(REFERENCE)
        self.assertEqual(grain_map.shape, ref['grain_map'].shape)
        np.testing.assert_array_equal(grain_map, ref['grain_map'])
        np.testing.assert_allclose(
            confidence_map, ref['confidence_map'], atol=1e-12)

    def test_spatially_adaptive_agrees_with_exhaustive(self):
        import nfutil
        from hexrd.hedm import nf_config

        configuration = nf_config.open_file(CONFIG)[0]
        experiment = nfutil.Experiment(configuration)
        image_stack = experiment.image_stack
        controller = nfutil.build_controller(configuration)
        Xs, Ys, Zs, mask, coords = experiment.generate_test_coordinates()
        precomputed = nfutil.precompute_diffraction_data(
            experiment, controller, experiment.exp_maps)
        _, conf_full, idx_full = nfutil.test_orientations_at_coordinates(
            experiment, controller, image_stack, precomputed, coords,
            refine_yes_no=0)
        _, conf, idx, stats = nfutil.reconstruct_spatially_adaptive(
            experiment, controller, image_stack, precomputed, coords, mask)

        # Far fewer orientation tests than the exhaustive search
        self.assertLess(stats['n_tests'], 0.4*stats['n_exhaustive_tests'])
        # Exhaustive is the optimum over the same candidates
        self.assertTrue((conf <= conf_full + 1e-12).all())
        # Several FF grains are near-duplicates, so compare confidence and
        # not grain ids: nearly every voxel is almost as good as the best
        close = conf >= conf_full - 0.05
        self.assertGreater(close.mean(), 0.97)
        self.assertLess(abs(conf.mean() - conf_full.mean()), 0.01)
        self.assertGreater((idx == idx_full).mean(), 0.85)

    def test_generate_experiment_wrapper(self):
        import nfutil
        from hexrd.hedm import nf_config

        configuration = nf_config.open_file(CONFIG)[0]
        experiment, image_stack = nfutil.generate_experiment(configuration)
        self.assertIsInstance(experiment, nfutil.Experiment)
        self.assertIs(image_stack, experiment.image_stack)
        self.assertEqual(experiment.nframes, image_stack.shape[0])

    def test_pickle_excludes_image_stack(self):
        import pickle
        import nfutil
        from hexrd.hedm import nf_config

        configuration = nf_config.open_file(CONFIG)[0]
        experiment = nfutil.Experiment(configuration)
        payload = pickle.dumps(experiment)
        # The stack is ~10 GB; the rest of the experiment is tiny
        self.assertLess(len(payload), 50 * 1024 * 1024)
        self.assertFalse(hasattr(pickle.loads(payload), 'image_stack'))


if __name__ == '__main__':
    unittest.main()
