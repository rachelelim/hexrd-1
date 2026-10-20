"""``hexrd tomo reconstruct`` on a synthetic scan: a disk on the rotation axis, radiographed at 90 angles."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tomo_testing as T
from hexrd.hedm import nf_config
from hexrd.hedm.grainmap.tomo_options import ReconstructOptions
from hexrd.hedm.grainmap.tomo_reconstruct import reconstruct, resolve_options


def iou(mask, ideal):
    return (mask & ideal).sum() / (mask | ideal).sum()


class Scan:
    """Files of one synthetic scan in a temporary folder."""

    def __init__(self, n_slots=90, dropped=(), **config):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.dark, self.bright, self.tomo = T.write_scan_files(self.dir, n_slots, dropped)
        self.detector = T.write_instrument(self.dir / 'det.yml')
        self.out = self.dir / 'out'
        self.n_slots = n_slots
        self.config = config

    def cfg(self, **more):
        kw = dict(num_images=self.n_slots, output_dir=self.out, **{**self.config, **more})
        path = T.write_config(self.dir / 'tomo.yml', self.detector, self.bright, self.dark, self.tomo, **kw)
        return nf_config.open_file(str(path))[0]

    def cleanup(self):
        self.tmp.cleanup()


class TestReconstruct(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scan = Scan()
        cls.path = Path(reconstruct(cls.scan.cfg(), ReconstructOptions(all_layers=True, processes=2)))
        cls.result = np.load(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.scan.cleanup()

    def test_mask_has_the_grid_and_layers_of_the_config(self):
        n = int(round(T.GRID_MM / T.VOXEL_MM))
        self.assertEqual(self.result['mask'].shape, (4, n, n))
        self.assertEqual(self.result['layer_indices'].tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(self.result['v_mm'], [-0.00675, -0.00225, 0.00225, 0.00675])
        self.assertAlmostEqual(float(self.result['voxel_spacing']), T.VOXEL_MM)

    def test_mask_is_the_disk(self):
        mask = self.result['mask']
        ideal = T.ideal_mask(mask.shape[1:])
        for layer in mask:
            self.assertGreater(iou(layer, ideal), 0.7)
        # the object does not change along z
        self.assertTrue(all(np.array_equal(mask[0], layer) for layer in mask))

    def test_output_name_and_no_temporary_files_left(self):
        self.assertEqual(self.path.name, 'synthetic_tomo_mask.npz')
        self.assertEqual(sorted(p.name for p in self.scan.out.iterdir()), ['synthetic_tomo_mask.npz'])

    def test_grid_coordinates_are_saved(self):
        for key in ('Xs', 'Ys', 'Zs'):
            self.assertEqual(self.result[key].shape[0], 4)

    def test_default_is_the_middle_layer(self):
        path = reconstruct(self.scan.cfg(), ReconstructOptions(processes=2, output_dir=str(self.scan.dir / 'one')))
        self.assertEqual(Path(path).name, 'synthetic_layer2_tomo_mask.npz')
        d = np.load(path)
        self.assertEqual(d['mask'].shape[0], 1)
        self.assertEqual(d['layer_indices'].tolist(), [2])

    def test_chosen_layers(self):
        path = reconstruct(self.scan.cfg(), ReconstructOptions(layer_indices=[0, 3], processes=2,
                                                               output_dir=str(self.scan.dir / 'two')))
        self.assertEqual(Path(path).name, 'synthetic_2layers_tomo_mask.npz')
        self.assertEqual(np.load(path)['layer_indices'].tolist(), [0, 3])

    def test_layer_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'outside 0..3'):
            reconstruct(self.scan.cfg(), ReconstructOptions(layer_indices=[9], processes=2))

    def test_save_fbp(self):
        out = self.scan.dir / 'fbp'
        reconstruct(self.scan.cfg(), ReconstructOptions(layer_index=1, save_fbp=True, processes=2, output_dir=str(out)))
        self.assertTrue((out / 'synthetic_layer1_fbp_layer1.npy').exists())


class TestDroppedFrames(unittest.TestCase):

    def test_angles_come_from_the_timestamps(self):
        scan = Scan(dropped=(10, 40, 41))
        try:
            mask = np.load(reconstruct(scan.cfg(), ReconstructOptions(all_layers=True, processes=2)))['mask']
            self.assertGreater(iou(mask[0], T.ideal_mask(mask.shape[1:])), 0.7)
        finally:
            scan.cleanup()


class TestYmlAndFlags(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # irregular timestamps: the angles cannot come from them, so they have to come from the yml or a flag
        cls.scan = Scan()
        frames, _ = T.tomo_frames(90, ())
        rng = np.random.default_rng(1)
        T.write_h5(cls.scan.tomo, frames, slots=np.cumsum(np.r_[0, rng.uniform(0.2, 0.8, 89)]) / T.PERIOD)
        cls.slots = cls.scan.dir / 'slots.txt'
        cls.slots.write_text('')                    # no dropped frames
        cls.theta = cls.scan.dir / 'theta.npy'
        np.save(cls.theta, T.angles(90)[0])

    @classmethod
    def tearDownClass(cls):
        cls.scan.cleanup()

    def mask(self, cfg, options):
        options.processes = 2
        options.output_dir = str(self.scan.dir / 'o')
        return np.load(reconstruct(cfg, options))['mask'][0]

    def test_irregular_timestamps_fail_without_angles(self):
        with self.assertRaisesRegex(RuntimeError, 'not an integer number'):
            reconstruct(self.scan.cfg(), ReconstructOptions(processes=2))

    def test_missing_slots_from_the_yml(self):
        cfg = self.scan.cfg(extra_tomography={'missing_slots': str(self.slots)})
        self.assertGreater(iou(self.mask(cfg, ReconstructOptions()), T.ideal_mask((12, 12))), 0.7)

    def test_theta_file_from_the_yml(self):
        cfg = self.scan.cfg(extra_tomography={'theta_file': str(self.theta)})
        self.assertGreater(iou(self.mask(cfg, ReconstructOptions()), T.ideal_mask((12, 12))), 0.7)

    def test_flag_overrides_the_yml(self):
        cfg = self.scan.cfg(extra_tomography={'missing_slots': str(self.scan.dir / 'does_not_exist.txt')})
        with self.assertRaises(FileNotFoundError):
            reconstruct(cfg, ReconstructOptions(processes=2))
        self.assertGreater(iou(self.mask(cfg, ReconstructOptions(missing_slots=str(self.slots))),
                               T.ideal_mask((12, 12))), 0.7)

    def test_resolve_options_fills_in_the_yml_values(self):
        cfg = self.scan.cfg(extra_tomography={'missing_slots': 'a.txt', 'theta_file': 'b.npy', 'deg_per_frame': 2.0,
                                              'spec': {'log': 's.log', 'scan': 5}})
        opts = resolve_options(cfg, ReconstructOptions())
        self.assertEqual((opts.missing_slots, opts.theta_file, opts.deg_per_frame), ('a.txt', 'b.npy', 2.0))
        self.assertEqual((opts.spec_log, opts.spec_scan), ('s.log', 5))
        self.assertEqual(opts.processes, 2)             # multiprocessing: num_cpus of the yml
        flags = resolve_options(cfg, ReconstructOptions(missing_slots='c.txt', deg_per_frame=1.0, processes=1,
                                                        spec_log='t.log', spec_scan=6))
        self.assertEqual((flags.missing_slots, flags.deg_per_frame, flags.processes), ('c.txt', 1.0, 1))
        self.assertEqual((flags.spec_log, flags.spec_scan), ('t.log', 6))

    def test_resolve_options_does_not_change_the_options_it_is_given(self):
        options = ReconstructOptions()
        resolve_options(self.scan.cfg(extra_tomography={'deg_per_frame': 2.0}), options)
        self.assertIsNone(options.deg_per_frame)

    def test_spec_log_needs_a_scan(self):
        with self.assertRaisesRegex(ValueError, 'together'):
            resolve_options(self.scan.cfg(), ReconstructOptions(spec_log='spec.log'))


class TestFailureCleanup(unittest.TestCase):

    def test_temporary_stack_is_removed_when_a_worker_fails(self):
        scan = Scan()
        try:
            cfg = scan.cfg()
            cfg._cfg['tomography']['processing']['recon_thresh'] = 'not a number'
            with self.assertRaises(TypeError):
                reconstruct(cfg, ReconstructOptions(all_layers=True, processes=2))
            self.assertEqual([p.name for p in scan.out.iterdir() if p.suffix == '.dat'], [])
        finally:
            scan.cleanup()


if __name__ == '__main__':
    unittest.main()
