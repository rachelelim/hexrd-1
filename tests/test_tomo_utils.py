"""Tests for the small tomography modules: spec log, masks, missing frames, the nf_config accessors
and the ``hexrd tomo`` parser. No experimental data is needed (see tomo_testing.py)."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tomo_testing as T
from hexrd.hedm import nf_config
from hexrd.hedm.cli import tomo as tomo_cli
from hexrd.hedm.grainmap import tomo_frames, tomo_masks, tomo_spec


class TestSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = T.write_spec_log(Path(cls.tmp.name) / 'spec.log', n_slots=20, dropped=(5, 6), numbers=(7, 8, 9))
        cls.scans = tomo_spec.parse_spec(cls.log)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scans_and_columns(self):
        self.assertEqual(sorted(self.scans), [7, 8, 9])
        s = self.scans[9]
        self.assertEqual(s.columns, ['ome', 'det_exp_out'])
        self.assertEqual(s.npoints, 20)
        self.assertEqual(self.scans[7].npoints, 5)

    def test_only_the_scan_after_the_note_is_tomo(self):
        self.assertEqual([n for n, s in self.scans.items() if s.is_tomo], [9])

    def test_retiga_file_of_each_scan(self):
        self.assertEqual(Path(self.scans[8].retiga_file).name, 'retiga_000008.h5')

    def test_dropped_slots_from_det_exp_out(self):
        self.assertEqual(tomo_spec.dropped_slots(self.scans[9]).tolist(), [5, 6])

    def test_ragged_last_row_is_dropped(self):
        path = Path(self.tmp.name) / 'ragged.log'
        path.write_text('#S 1  scan\n#L a  b\n1 2\n3 4\n5\n')
        s = tomo_spec.parse_spec(path)[1]
        self.assertEqual(s.npoints, 2)

    def test_write_missing_slots(self):
        written = tomo_spec.write_missing_slots(self.scans, Path(self.tmp.name) / 'slots')
        self.assertEqual([(n, k) for n, k, _ in written], [(9, 2)])
        self.assertEqual(Path(written[0][2]).read_text().split(), ['5', '6'])

    def test_text_summaries(self):
        self.assertIn('1 tomography scans', tomo_spec.tomo_summary(self.scans)[0])
        self.assertEqual(len(tomo_spec.scan_table(self.scans)), 4)
        self.assertTrue(any('rams pulse step' in line for line in tomo_spec.scan_details(self.scans[9])))


class TestMasks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        yy, xx = np.mgrid[:40, :40]
        disk = (yy - 20) ** 2 + (xx - 20) ** 2 <= 100
        self.mask = np.stack([disk] * 10)
        self.mask[4:6] = False                  # two bad layers
        self.src = Path(self.tmp.name) / 'mask.npz'
        np.savez_compressed(self.src, mask=self.mask, voxel_spacing=0.01)

    def tearDown(self):
        self.tmp.cleanup()

    def test_signed_distance_is_positive_inside(self):
        s = tomo_masks.sdf(self.mask[0])
        self.assertGreater(s[20, 20], 5)
        self.assertLess(s[0, 0], 0)

    def test_interpolation_fills_bad_layers(self):
        dst = Path(self.tmp.name) / 'filled.npz'
        tomo_masks.interpolate_mask_file(self.src, dst, [(4, 5)])
        d = np.load(dst)
        self.assertTrue(np.array_equal(d['mask'][4], self.mask[3]))
        self.assertTrue(np.array_equal(d['mask'][5], self.mask[3]))
        self.assertEqual(d['interpolated_layers'].tolist(), [4, 5])
        self.assertEqual(json.loads(str(d['interpolation_support'])), {'4-5': [2, 3, 6, 7]})
        self.assertTrue(np.array_equal(d['mask'][:4], self.mask[:4]))

    def test_interpolation_needs_support_inside_the_mask(self):
        with self.assertRaisesRegex(ValueError, 'needs layers'):
            tomo_masks.interpolate_mask_file(self.src, Path(self.tmp.name) / 'x.npz', [(0, 1)])

    def test_interpolation_ranges_may_not_share_support(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            tomo_masks.interpolate_mask_file(self.src, Path(self.tmp.name) / 'x.npz', [(3, 3), (5, 5)])

    def test_parse_ranges(self):
        self.assertEqual(tomo_masks.parse_ranges('15-17,28-31,63'), [(15, 17), (28, 31), (63, 63)])

    def test_smoothing_keeps_the_keys_and_the_shape(self):
        dst = Path(self.tmp.name) / 'smooth.npz'
        volume, changed = tomo_masks.smooth_mask_file(self.src, dst, 1.5)
        d = np.load(dst)
        self.assertEqual(set(d.files), {'mask', 'voxel_spacing', 'mask_unsmoothed', 'sdf_smooth', 'smooth_sigma_voxels'})
        self.assertTrue(np.array_equal(d['mask_unsmoothed'], self.mask))
        self.assertEqual(d['mask'].shape, self.mask.shape)
        self.assertLess(changed, 25)

    def test_smoothing_a_disk_changes_little(self):
        new, s = tomo_masks.smooth(self.mask[:1], 1.0)
        self.assertGreater((new & self.mask[:1]).sum() / self.mask[:1].sum(), 0.9)
        self.assertEqual(s.dtype, np.float32)

    def test_per_layer_smoothing_does_not_mix_layers(self):
        new, _ = tomo_masks.smooth(self.mask, 1.0, per_layer=True)
        self.assertFalse(new[4].any())


class TestFrames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stack(self, name, n_slots, dropped):
        frames, slots = T.tomo_frames(n_slots, dropped)
        path = self.folder / name
        T.write_h5(path, frames, slots)
        return path

    def test_no_missing_frames(self):
        path = self.stack('full.h5', 30, ())
        found = tomo_frames.find_missing_frames(path, deg_per_slot=2.0, plot=False)
        self.assertEqual(found.n_missing, 0)
        self.assertEqual(found.slot.tolist(), list(range(30)))
        np.testing.assert_allclose(found.theta, 2.0 * np.arange(30))

    def test_dropped_frames_are_found_and_the_angles_skip_them(self):
        path = self.stack('dropped.h5', 30, (7, 8, 20))
        found = tomo_frames.find_missing_frames(path, deg_per_slot=2.0, expected=30, plot=False)
        self.assertEqual(found.missing.tolist(), [7, 8, 20])
        self.assertEqual(len(found.slot), 27)
        np.testing.assert_allclose(found.theta, 2.0 * found.slot)
        prefix = path.with_suffix('')
        self.assertEqual(np.loadtxt(f'{prefix}_missing_slots.txt', dtype=int).tolist(), [7, 8, 20])
        self.assertEqual(np.load(f'{prefix}_theta_deg.npy').shape, (27,))
        self.assertTrue(Path(f'{prefix}_frame_angles.csv').exists())

    def test_span_sets_the_step(self):
        path = self.stack('span.h5', 30, ())
        found = tomo_frames.find_missing_frames(path, span_deg=180, plot=False)
        self.assertAlmostEqual(found.theta[-1], 180.0)

    def test_untriggered_stack_is_an_error(self):
        frames, _ = T.tomo_frames(12, ())
        path = self.folder / 'irregular.h5'
        T.write_h5(path, frames, slots=np.cumsum(np.r_[0, np.random.default_rng(0).uniform(0.2, 0.8, 11)]) / T.PERIOD)
        with self.assertRaisesRegex(RuntimeError, 'not close to an integer'):
            tomo_frames.find_missing_frames(path, plot=False)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            tomo_frames.find_missing_frames(self.folder / 'nope.h5', plot=False)


class TestTomographyConfig(unittest.TestCase):

    def cfg(self, **tomography):
        return nf_config.nf_root.NFRootConfig({'tomography': tomography, 'multiprocessing': {'num_cpus': 3}})

    def test_new_keys_default_to_none(self):
        tomo = self.cfg().tomography
        self.assertIsNone(tomo.spec.log)
        self.assertIsNone(tomo.spec.scan)
        self.assertIsNone(tomo.spec.tomo_dir)
        self.assertIsNone(tomo.spec.sample_dir)
        self.assertIsNone(tomo.missing_slots)
        self.assertIsNone(tomo.theta_file)
        self.assertIsNone(tomo.deg_per_frame)

    def test_new_keys_are_read(self):
        tomo = self.cfg(spec={'log': 'spec.log', 'scan': 23, 'tomo_dir': '/data', 'sample_dir': '/sample'},
                        missing_slots='slots.txt',
                        theta_file='theta.npy', deg_per_frame=0.25).tomography
        self.assertEqual((tomo.spec.log, tomo.spec.scan, tomo.spec.tomo_dir, tomo.spec.sample_dir),
                         ('spec.log', 23, '/data', '/sample'))
        self.assertEqual((tomo.missing_slots, tomo.theta_file, tomo.deg_per_frame), ('slots.txt', 'theta.npy', 0.25))


class TestParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        tomo_cli.configure_parser(cls.parser.add_subparsers(dest='cmd'))

    def parse(self, *argv):
        with contextlib.redirect_stderr(io.StringIO()):     # argparse prints its usage on an error
            return self.parser.parse_args(['tomo', *argv])

    def test_every_step_is_registered_with_its_function(self):
        cases = {'reconstruct': ['c.yml'], 'pipeline': ['c.yml'], 'missing-frames': ['a.h5'], 'spec': ['s.log'],
                 'interpolate': ['m.npz', 'o.npz', '3-4'], 'smooth': ['m.npz', 'o.npz', '1.5']}
        for step, argv in cases.items():
            args = self.parse(step, *argv)
            self.assertEqual(args.tomo_step, step)
            self.assertTrue(callable(args.func), step)

    def test_a_step_is_required(self):
        with self.assertRaises(SystemExit):
            self.parse()

    def test_reconstruct_leaves_yml_values_unset(self):
        args = self.parse('reconstruct', 'c.yml')
        for name in ('processes', 'theta_file', 'missing_slots', 'spec_log', 'spec_scan', 'deg_per_frame',
                     'layer_index', 'layer_indices', 'vert_beam_center', 'max_angle'):
            self.assertIsNone(getattr(args, name), name)
        self.assertEqual(args.powder_open_um, 0.0)
        self.assertEqual(args.row_margin, 16)

    def test_reconstruct_flags(self):
        args = self.parse('reconstruct', 'c.yml', '--layer-indices', '1', '3', '--spec-log', 's.log', '--spec-scan',
                          '23', '--powder-open-um', '40', '--processes', '4', '-q')
        self.assertEqual((args.layer_indices, args.spec_log, args.spec_scan), ([1, 3], 's.log', 23))
        self.assertEqual((args.powder_open_um, args.processes, args.quiet), (40.0, 4, True))

    def test_pipeline_defaults_match_the_options(self):
        from hexrd.hedm.grainmap.tomo_options import STAGES, PipelineOptions
        args = self.parse('pipeline', 'c.yml')
        self.assertEqual(args.stages, ','.join(STAGES))
        d = PipelineOptions()
        self.assertEqual((args.smooth_sigma, args.vert_beam_center, args.powder_open_um),
                         (d.smooth_sigma, d.vert_beam_center, d.powder_open_um))
        self.assertIsNone(args.processes)

    def test_interpolate_support_needs_four_layers(self):
        self.assertEqual(self.parse('interpolate', 'm', 'o', '5-6', '--support', '3', '4', '7', '8').support,
                         [3, 4, 7, 8])
        with self.assertRaises(SystemExit):
            self.parse('interpolate', 'm', 'o', '5-6', '--support', '3', '4')

    def test_smooth_per_layer_flag(self):
        self.assertTrue(self.parse('smooth', 'm', 'o', '1', '--2d').per_layer)
        self.assertFalse(self.parse('smooth', 'm', 'o', '1').per_layer)

    def test_building_the_parser_does_not_import_the_heavy_modules(self):
        import subprocess
        import sys
        code = ("import sys, argparse\n"
                "from hexrd.hedm.cli import tomo\n"
                "tomo.configure_parser(argparse.ArgumentParser().add_subparsers())\n"
                "print([m for m in ('hexrd.hedm.grainmap.tomo_reconstruct', 'hexrd.hedm.grainmap.nfutil') "
                "if m in sys.modules])\n")
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip().splitlines()[-1], '[]')


class TestSteps(unittest.TestCase):
    """The steps that need no reconstruction, run through the same execute functions as the command."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.parser = argparse.ArgumentParser()
        tomo_cli.configure_parser(self.parser.add_subparsers(dest='cmd'))

    def tearDown(self):
        self.tmp.cleanup()

    def run_step(self, *argv):
        args = self.parser.parse_args(['tomo', *argv])
        args.debug = False
        args.func(args, self.parser)

    def test_smooth_step_writes_the_file(self):
        yy, xx = np.mgrid[:30, :30]
        mask = np.stack([(yy - 15) ** 2 + (xx - 15) ** 2 <= 64] * 6)
        np.savez_compressed(self.folder / 'm.npz', mask=mask, voxel_spacing=0.01)
        self.run_step('smooth', str(self.folder / 'm.npz'), str(self.folder / 'o.npz'), '1.0', '-q')
        self.assertIn('sdf_smooth', np.load(self.folder / 'o.npz').files)

    def test_interpolate_step_reports_a_bad_range_and_exits(self):
        mask = np.ones((6, 8, 8), bool)
        np.savez_compressed(self.folder / 'm.npz', mask=mask)
        with self.assertRaises(SystemExit):
            self.run_step('interpolate', str(self.folder / 'm.npz'), str(self.folder / 'o.npz'), '0-1', '-q')

    def test_missing_frames_step_exits_2_on_a_missing_file(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_step('missing-frames', str(self.folder / 'nope.h5'), '-q')
        self.assertEqual(ctx.exception.code, 2)

    def test_spec_step_lists_the_scans(self):
        log = T.write_spec_log(self.folder / 'spec.log', n_slots=10, numbers=(1, 2, 3))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.run_step('spec', str(log))
            self.run_step('spec', str(log), '3')
            self.run_step('spec', str(log), '--tomo')
        text = out.getvalue()
        self.assertIn('3 scans', text)
        self.assertIn('Scan 3:', text)
        self.assertIn('1 tomography scans', text)


if __name__ == '__main__':
    unittest.main()
