"""Regression tests: single-layer NF reconstructions of Ti-7Al (ti7-1).

Two datasets, with and without a tomography mask. Each runs the same steps as
nf_hedm/reconstruct_nf.py in-process (without saving) and compares the grain
and confidence maps against a stored reference, then compares the spatially
adaptive reconstruction against testing every FF grain at every voxel.

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
    # Spatially adaptive thresholds: fraction of the all-grains orientation
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

    def test_spatially_adaptive_agrees_with_all_grains(self):
        import nfutil

        _, conf, idx, stats = nfutil.reconstruct_spatially_adaptive(
            self.experiment, self.controller, self.experiment.image_stack,
            self.precomputed, self.coords, self.mask)

        # Far fewer orientation tests than testing every grain at every voxel
        self.assertLess(stats['n_tests'],
                        self.max_test_fraction*stats['n_all_grains_tests'])
        # Testing every grain is the optimum over the same candidates
        self.assertTrue((conf <= self.raw_confidence + 1e-12).all())
        # Several FF grains are near-duplicates, so compare confidence and
        # not grain ids: nearly every voxel is almost as good as the best
        close = conf >= self.raw_confidence - 0.05
        self.assertGreater(close.mean(), self.min_close)
        self.assertLess(abs(conf.mean() - self.raw_confidence.mean()), 0.01)
        self.assertGreater((idx == self.raw_idx).mean(), self.min_same_grain)


    def test_refinement_along_the_marching_agrees_with_refining_each_voxel(self):
        import nfutil
        from nfutil.orientation_search import misorientation_deg

        exp_maps, conf, origin, misorientation, stats = \
            nfutil.reconstruct_and_refine_spatially_adaptive(
                self.experiment, self.controller, self.experiment.image_stack,
                self.precomputed, self.coords, self.mask)

        # Refinement fits the images better than the grain orientations do
        self.assertGreater(conf.mean(), self.raw_confidence.mean() + 0.03)
        # and takes far fewer simulations than refining every voxel from its
        # grain, which takes about 314 per voxel. The march alone takes about
        # half of that, and the two boundary passes add to it (more on the small
        # unmasked grid, where a larger share of voxels are at a boundary).
        self.assertLess(stats['n_simulated']/stats['n_voxels'], 0.7*314)
        # Orientations move by a fraction of a degree, as when each voxel is refined alone
        self.assertLess(np.median(misorientation), 0.25)

        # Against refining a sample of voxels each on its own. The sample has more voxels than
        # there are grains, so test_orientations_at_coordinates splits the work over voxels,
        # as it does for a whole layer. With fewer it splits over grains and refines the best
        # of each, which is a different (slower) method.
        sample = np.random.default_rng(4).choice(len(self.coords), 800, replace=False)
        alone_em, alone_conf, _, _ = nfutil.test_orientations_at_coordinates(
            self.experiment, self.controller, self.experiment.image_stack, self.precomputed,
            self.coords[sample], refine_yes_no=1, return_misorientation=1)
        alone_conf = np.asarray(alone_conf).ravel()
        # Not worse overall, and few voxels worse by much. On the full layers 97 to 99%
        # are within 0.05, and a sample of 800 varies by about a percent.
        self.assertGreater(conf[sample].mean(), alone_conf.mean() - 0.01)
        self.assertGreater((conf[sample] >= alone_conf - 0.05).mean(), 0.94)
        # Where they land on the same orientation they agree to a fraction of a
        # degree. About 7% of voxels (the weakly determined ones, where another
        # grain fits as well) end on different grains, and neither is better
        # on average, so only that many are expected to differ.
        q_sym = self.experiment.plane_data.q_sym
        difference = np.array([misorientation_deg(alone_em[i], exp_maps[v][None], q_sym)[0]
                               for i, v in enumerate(sample)])
        self.assertGreater((difference < 0.3).mean(), 0.88)


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

    def _known_voxels(self, n, seed, low=0.8):
        """Voxels the reference gives to an FF grain with confidence above
        ``low``, with that grain's orientation. The FF orientations come from
        far-field data, so they are independent of the NF images."""
        ref = np.load(os.path.join(HERE, 'data', self.reference))
        m = self.mask.astype(bool)
        grain = ref['grain_map'][m].astype(int)
        conf = ref['confidence_map'][m]
        rng = np.random.default_rng(seed)
        voxels = rng.choice(np.where(conf > low)[0], n, replace=False)
        truth = np.array([self.experiment.exp_maps[
            np.where(self.experiment.remap == grain[v])[0][0]] for v in voxels])
        return voxels, truth, conf[voxels]

    def test_orientation_search_recovers_ff_orientations(self):
        import nfutil
        from nfutil.orientation_search import misorientation_deg

        voxels, truth, ff_conf = self._known_voxels(16, seed=3)
        exp_maps, conf, stats = nfutil.search_orientations(
            self.experiment, self.controller, self.experiment.image_stack,
            self.coords[voxels])
        q_sym = self.experiment.plane_data.q_sym
        mis = np.array([misorientation_deg(t, e[None], q_sym)[0]
                        for t, e in zip(truth, exp_maps)])
        # Held-out checks of this search found about 96% within 0.5 degrees
        self.assertGreaterEqual((mis < 0.5).mean(), 0.85)
        # Where it did recover the orientation, it fits the images about as
        # well as the FF grain does
        found = mis < 0.5
        self.assertTrue((conf[found] >= ff_conf[found] - 0.05).all())

    @unittest.skipUnless(os.environ.get('NF_SLOW_TESTS'),
                         'takes about 3 minutes, set NF_SLOW_TESTS=1')
    def test_find_missing_orientations_leave_out(self):
        """Remove grains from the FF list, then find them again."""
        import nfutil
        from nfutil.orientation_search import misorientation_deg

        # A separate experiment, since grains are removed from it
        experiment = nfutil.Experiment(self.configuration)
        ref = np.load(os.path.join(HERE, 'data', self.reference))
        m = self.mask.astype(bool)
        grain = ref['grain_map'][m].astype(int)
        ids, counts = np.unique(grain, return_counts=True)
        rng = np.random.default_rng(5)
        removed_ids = rng.choice(ids[counts >= 300], 6, replace=False)
        removed = np.array([np.where(experiment.remap == i)[0][0] for i in removed_ids])
        truth = experiment.exp_maps[removed].copy()
        keep = np.setdiff1d(np.arange(len(experiment.exp_maps)), removed)
        experiment.exp_maps = experiment.exp_maps[keep]
        experiment.remap = experiment.remap[keep]
        experiment.n_grains = len(keep)
        image_stack = experiment.image_stack
        data = nfutil.precompute_diffraction_data(
            experiment, self.controller, experiment.exp_maps)
        _, before, _, _ = nfutil.reconstruct_spatially_adaptive(
            experiment, self.controller, image_stack, data, self.coords, self.mask)

        found = nfutil.find_missing_orientations(
            experiment, self.controller, image_stack, data, self.coords, self.mask)
        q_sym = experiment.plane_data.q_sym
        recovered = sum(misorientation_deg(t, found['exp_maps'], q_sym).min() < 0.5
                        for t in truth) if len(found['exp_maps']) else 0
        self.assertGreaterEqual(recovered, 4)    # 5 of 6 in the check this is based on

        data = nfutil.add_orientations(experiment, data, found['exp_maps'])
        _, after, _, _ = nfutil.reconstruct_spatially_adaptive(
            experiment, self.controller, image_stack, data, self.coords, self.mask)
        gone = np.isin(grain, removed_ids)
        self.assertGreater(after[gone].mean(), before[gone].mean() + 0.15)
        self.assertLess((after < 0.5).sum(), 0.5*(before < 0.5).sum())


if __name__ == '__main__':
    unittest.main()
