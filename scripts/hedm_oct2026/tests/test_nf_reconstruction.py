"""Regression tests: single-layer NF reconstructions of Ti-7Al (ti7-1).

Two datasets, with and without a tomography mask. Each runs the same steps as
nf_hedm/reconstruct_nf.py in-process (without saving) and compares the grain
and confidence maps against a stored reference, then compares the spatially
adaptive reconstruction against the exhaustive one.

The references record what the code produced when they were saved. They catch
unintended changes, they are not a measurement of the true microstructure.

The data folder is read from NF_TEST_DATA_DIR (default below). The tests are
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
    '/Users/lim37/Library/CloudStorage/OneDrive-LLNL/ti7al_cyclic_HEDM/'
    'ti7-1/NF')
HAVE_DATA = (os.path.isfile(os.path.join(DATA_DIR, 'S0_L0.yml'))
             and os.path.isfile(os.path.join(
                 DATA_DIR, '..', 'Ti7Al_init_tomo_clean.npz')))


class ReconstructionCases:
    """Tests shared by the datasets. Subclasses set the attributes below."""

    # Config file (relative to the data folder, or absolute) and reference
    config = None
    reference = None
    # Spatially adaptive thresholds: fraction of the exhaustive orientation
    # tests it may use, fraction of voxels within 0.05 of the best confidence,
    # and fraction of voxels that must pick the same grain
    max_test_fraction = None
    min_close = None
    min_same_grain = None

    @classmethod
    def setUpClass(cls):
        import nfutil
        from hexrd.hedm import nf_config

        # The configs use paths relative to the data folder
        cls._cwd = os.getcwd()
        os.chdir(DATA_DIR)
        cls.addClassCleanup(os.chdir, cls._cwd)

        cls.configuration = nf_config.open_file(cls.config)[0]
        cls.experiment = nfutil.Experiment(cls.configuration)
        cls.controller = nfutil.build_controller(cls.configuration)
        (cls.Xs, cls.Ys, cls.Zs, cls.mask,
         cls.coords) = cls.experiment.generate_test_coordinates()
        cls.precomputed = nfutil.precompute_diffraction_data(
            cls.experiment, cls.controller, cls.experiment.exp_maps)
        _, cls.raw_confidence, cls.raw_idx = \
            nfutil.test_orientations_at_coordinates(
                cls.experiment, cls.controller, cls.experiment.image_stack,
                cls.precomputed, cls.coords, refine_yes_no=0)
        cls.grain_map, cls.confidence_map = nfutil.process_raw_data(
            cls.raw_confidence, cls.raw_idx, cls.Xs.shape, mask=cls.mask,
            id_remap=cls.experiment.remap)

    def test_matches_reference(self):
        ref = np.load(os.path.join(HERE, 'data', self.reference))
        self.assertEqual(self.grain_map.shape, ref['grain_map'].shape)
        np.testing.assert_array_equal(self.grain_map, ref['grain_map'])
        np.testing.assert_allclose(
            self.confidence_map, ref['confidence_map'], atol=1e-12)

    def test_spatially_adaptive_agrees_with_exhaustive(self):
        import nfutil

        _, conf, idx, stats = nfutil.reconstruct_spatially_adaptive(
            self.experiment, self.controller, self.experiment.image_stack,
            self.precomputed, self.coords, self.mask)

        # Far fewer orientation tests than the exhaustive search
        self.assertLess(stats['n_tests'],
                        self.max_test_fraction*stats['n_exhaustive_tests'])
        # Exhaustive is the optimum over the same candidates
        self.assertTrue((conf <= self.raw_confidence + 1e-12).all())
        # Several FF grains are near-duplicates, so compare confidence and
        # not grain ids: nearly every voxel is almost as good as the best
        close = conf >= self.raw_confidence - 0.05
        self.assertGreater(close.mean(), self.min_close)
        self.assertLess(abs(conf.mean() - self.raw_confidence.mean()), 0.01)
        self.assertGreater((idx == self.raw_idx).mean(), self.min_same_grain)


@unittest.skipUnless(HAVE_DATA, f'NF test data not found in {DATA_DIR}')
class TestTi7AlUnmasked(ReconstructionCases, unittest.TestCase):
    """619 FF grains, hexagonal, 125x125 grid with no tomo mask."""
    config = 'S0_L0.yml'
    reference = 'ti7al_S0_L0_reference.npz'
    max_test_fraction = 0.8
    min_close = 0.98
    min_same_grain = 0.95

    def test_generate_experiment_wrapper(self):
        import nfutil

        experiment, image_stack = nfutil.generate_experiment(
            self.configuration)
        self.assertIsInstance(experiment, nfutil.Experiment)
        self.assertIs(image_stack, experiment.image_stack)
        self.assertEqual(experiment.nframes, image_stack.shape[0])

    def test_pickle_excludes_image_stack(self):
        import pickle

        payload = pickle.dumps(self.experiment)
        # The stack is ~6 GB; the rest of the experiment is tiny
        self.assertLess(len(payload), 50 * 1024 * 1024)
        self.assertFalse(hasattr(pickle.loads(payload), 'image_stack'))


@unittest.skipUnless(HAVE_DATA, f'NF test data not found in {DATA_DIR}')
class TestTi7AlMasked(ReconstructionCases, unittest.TestCase):
    """The same sample with the tomo mask: one 280x280 layer at 0.005 mm,
    39,762 masked voxels."""
    config = os.path.join(HERE, 'data', 'ti7al_S0_L0_masked.yml')
    reference = 'ti7al_S0_L0_masked_reference.npz'
    max_test_fraction = 0.6
    min_close = 0.99
    min_same_grain = 0.97


if __name__ == '__main__':
    unittest.main()
