"""The vertical stage position (ramsz) that ties a tomography mask to an NF layer: read from the spec log, saved in
the mask, and turned into the offset the NF reconstruction adds to the mask heights."""
import contextlib
import io
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tomo_testing as T
from hexrd.hedm import nf_config
from hexrd.hedm.cli import tomo as tomo_cli
from hexrd.hedm.grainmap import tomo_spec
from hexrd.hedm.grainmap.nfutil.utilities import generate_test_coordinates, resolve_vertical_motor_position
from hexrd.hedm.grainmap.tomo_options import ReconstructOptions
from hexrd.hedm.grainmap.tomo_reconstruct import reconstruct

# ramsz is logged in a #C comment just before the scans at a position, as in the real log
SPEC = """#F spec.log
#C Wed Sep 30 18:49:36 2026.  ramsz reset from -0.0500001 to 0.
#S 1  flyscan ome 0 10 20 0.2
#L ome  det_exp_out
0.1  0.2
#C Wed Sep 30 21:31:02 2026.  line 1 ramxx 6.10352e-06 ramsz -0.35.
#S 2  flyscan ome 0 180 720 0.2
#L ome  det_exp_out
0.1  0.2
#C Wed Sep 30 21:36:05 2026.  line 2 ramxx 6.10352e-06 ramsz 0.35.
#S 3  flyscan ome 0 180 720 0.2
#L ome  det_exp_out
0.1  0.2
#S 4  flyscan ome 0 90 450 0.5
#L ome  det_exp_out
0.1  0.2
#C Wed Sep 30 21:46:04 2026.  line 9 ramsx 6.10352e-06 ramsz -0.025.
#S 5  flyscan ome 0 90 450 0.5
#L ome  det_exp_out
0.1  0.2
"""
NO_RAMSZ = "#F spec.log\n#S 1  flyscan ome 0 10 20 0.2\n#L ome  det_exp_out\n0.1  0.2\n"


def write_mask(path, ramsz=0.35, n_layers=7, spacing=0.1):
    """A small mask of n_layers heights centered on 0 (a tomography scan at ``ramsz``)."""
    v = (np.arange(n_layers) - (n_layers - 1) / 2.0) * spacing
    shape = (n_layers, 3, 3)
    ys = np.broadcast_to(v[:, None, None], shape).copy()
    arrays = dict(mask=np.ones(shape, bool), Xs=np.zeros(shape), Ys=ys, Zs=np.zeros(shape),
                  voxel_spacing=spacing, layer_indices=np.arange(n_layers), v_mm=v)
    if ramsz is not None:
        arrays['ramsz'] = ramsz
    np.savez_compressed(path, **arrays)
    return path


class TestRamszFromTheSpecLog(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / 'spec.log'
        self.log.write_text(SPEC)

    def tearDown(self):
        self.tmp.cleanup()

    def test_value_in_effect_at_each_scan(self):
        expected = {1: 0.0, 2: -0.35, 3: 0.35, 4: 0.35, 5: -0.025}      # 4 inherits from the line before it
        self.assertEqual({n: tomo_spec.ramsz_of_scan(self.log, n) for n in expected}, expected)

    def test_parse_spec_attaches_it_to_the_scans(self):
        scans = tomo_spec.parse_spec(self.log)
        self.assertEqual({n: s.ramsz for n, s in scans.items()}, {1: 0.0, 2: -0.35, 3: 0.35, 4: 0.35, 5: -0.025})

    def test_unknown_before_the_first_comment(self):
        self.log.write_text(NO_RAMSZ)
        self.assertIsNone(tomo_spec.ramsz_of_scan(self.log, 1))
        self.assertIsNone(tomo_spec.parse_spec(self.log)[1].ramsz)

    def test_scan_not_in_the_log(self):
        with self.assertRaisesRegex(ValueError, 'scan 9 not found'):
            tomo_spec.ramsz_of_scan(self.log, 9)

    def test_offset_is_nf_minus_tomo(self):
        z_tomo, z_nf, offset = tomo_spec.vertical_offset(self.log, 3, 5)
        self.assertEqual((z_tomo, z_nf), (0.35, -0.025))
        self.assertAlmostEqual(offset, -0.375)
        # a layer with the sample higher than the tomography scan has a positive offset
        self.assertGreater(tomo_spec.vertical_offset(self.log, 2, 3)[2], 0)

    def test_offset_needs_a_position_for_both_scans(self):
        self.log.write_text(NO_RAMSZ + SPEC.split('#S 1', 1)[1].replace('#S 2', '#S 12').replace('#S 3', '#S 13'))
        # scan 1 has no position (nothing logged before it); scan 12 follows a ramsz comment
        self.assertIsNone(tomo_spec.ramsz_of_scan(self.log, 1))
        with self.assertRaisesRegex(ValueError, r'no vertical position \(ramsz\) for scan\(s\) \[1\]'):
            tomo_spec.vertical_offset(self.log, 1, 12)
        with self.assertRaisesRegex(ValueError, r'scan\(s\) \[1\]'):
            tomo_spec.vertical_offset(self.log, 1, 1)


class TestMaskRecordsRamsz(unittest.TestCase):

    def reconstruct(self, spec_ramsz=None, config_ramsz=None, use_spec=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        dark, bright, tomo = T.write_scan_files(d, 90)
        det = T.write_instrument(d / 'det.yml')
        log = T.write_spec_log(d / 'spec.log', 90, numbers=(1, 2, 3), ramsz=spec_ramsz)
        extra = {} if config_ramsz is None else {'reconstruction': None}
        path = T.write_config(d / 'tomo.yml', det, bright, dark, tomo, output_dir=d / 'out')
        if config_ramsz is not None:
            import yaml
            c = yaml.safe_load(Path(path).read_text())
            c['tomography']['reconstruction']['vertical_motor_position'] = config_ramsz
            Path(path).write_text(yaml.safe_dump(c))
        cfg = nf_config.open_file(str(path))[0]
        opts = ReconstructOptions(layer_index=1, processes=2, spec_log=str(log) if use_spec else None,
                                  spec_scan=3 if use_spec else None)
        return np.load(reconstruct(cfg, opts))

    def test_from_the_spec_log(self):
        d = self.reconstruct(spec_ramsz=0.35)
        self.assertEqual((float(d['ramsz']), str(d['ramsz_source'])), (0.35, 'spec log'))

    def test_from_the_config_when_the_spec_log_has_none(self):
        d = self.reconstruct(config_ramsz=0.2)
        self.assertEqual((float(d['ramsz']), str(d['ramsz_source'])), (0.2, 'config'))
        d = self.reconstruct(config_ramsz=-0.1, use_spec=False)
        self.assertEqual((float(d['ramsz']), str(d['ramsz_source'])), (-0.1, 'config'))

    def test_the_spec_log_comes_first(self):
        d = self.reconstruct(spec_ramsz=0.35, config_ramsz=0.2)
        self.assertEqual((float(d['ramsz']), str(d['ramsz_source'])), (0.35, 'spec log'))

    def test_unknown_is_recorded_as_nan_with_a_warning(self):
        with self.assertLogs('hexrd', level='WARNING') as logs:
            d = self.reconstruct()
        self.assertTrue(np.isnan(float(d['ramsz'])))
        self.assertEqual(str(d['ramsz_source']), 'unknown')
        self.assertIn('vertical stage position (ramsz)', '\n'.join(logs.output))


class TestNfOffset(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / 'spec.log'
        self.log.write_text(SPEC)
        self.mask = write_mask(self.dir / 'mask.npz', ramsz=0.35)

    def tearDown(self):
        self.tmp.cleanup()

    def test_from_the_spec_log(self):
        offset, source = resolve_vertical_motor_position(self.mask, self.log, 5)
        self.assertAlmostEqual(offset, -0.375)
        self.assertEqual(source, 'spec log')

    def test_spec_log_wins_over_the_config_value_and_warns_if_they_differ(self):
        with self.assertLogs('hexrd', level='WARNING') as logs:
            offset, source = resolve_vertical_motor_position(self.mask, self.log, 5, configured=-1.0)
        self.assertEqual((round(offset, 6), source), (-0.375, 'spec log'))
        self.assertIn('differs from the spec log', '\n'.join(logs.output))

    def test_agreeing_config_value_does_not_warn(self):
        logger = logging.getLogger('hexrd')
        with self.assertNoLogs(logger, level='WARNING'):
            resolve_vertical_motor_position(self.mask, self.log, 5, configured=-0.375)

    def test_config_value_when_there_is_no_spec_log(self):
        self.assertEqual(resolve_vertical_motor_position(self.mask, configured=-0.3), (-0.3, 'config'))
        self.assertEqual(resolve_vertical_motor_position(self.mask, None, None, 0), (0.0, 'config'))

    def test_config_value_when_the_mask_has_no_ramsz(self):
        old = write_mask(self.dir / 'old.npz', ramsz=None)
        with self.assertLogs('hexrd', level='WARNING') as logs:
            result = resolve_vertical_motor_position(old, self.log, 5, configured=-0.3)
        self.assertEqual(result, (-0.3, 'config'))
        self.assertIn('does not record', '\n'.join(logs.output))

    def test_config_value_when_the_scan_has_no_ramsz(self):
        log = self.dir / 'bare.log'
        log.write_text(NO_RAMSZ)
        with self.assertLogs('hexrd', level='WARNING') as logs:
            result = resolve_vertical_motor_position(self.mask, log, 1, configured=0.1)
        self.assertEqual(result, (0.1, 'config'))
        self.assertIn('no vertical position', '\n'.join(logs.output))

    def test_nothing_known_is_zero_with_a_warning(self):
        with self.assertLogs('hexrd', level='WARNING'):
            self.assertEqual(resolve_vertical_motor_position(self.mask), (0.0, 'none'))

    def test_sign_sample_higher_in_the_nf_layer_moves_the_mask_up(self):
        """Mask at ramsz 0.35; an NF layer at 0.45 is 0.1 mm higher, so the mask feature at v = 0 sits at +0.1."""
        self.log.write_text(SPEC.replace('ramsz -0.025.', 'ramsz 0.45.'))
        offset, _ = resolve_vertical_motor_position(self.mask, self.log, 5)
        self.assertAlmostEqual(offset, 0.1)

    def test_selected_layers_follow_the_offset(self):
        # layers at v = -0.3 ... +0.3 (0.1 apart), mask at ramsz 0.35. NF scan 5 is at -0.025, offset -0.375:
        # NF heights -0.675 ... -0.075, so a span around 0 matches nothing, a span around -0.4 does
        offset, _ = resolve_vertical_motor_position(self.mask, self.log, 5)
        kw = dict(voxel_spacing=0.1, mask_data_file=self.mask, vertical_motor_position=offset)
        ys = generate_test_coordinates([-0.025, 0.025], **kw)[1]
        self.assertEqual(ys.size, 0)
        ys = generate_test_coordinates([-0.45, -0.35], **kw)[1]
        np.testing.assert_allclose(np.unique(ys), [-0.375], atol=1e-9)    # the layer at v = 0, shifted to -0.375


class TestOffsetCommand(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / 'spec.log'
        self.log.write_text(SPEC)
        self.mask = write_mask(self.dir / 'mask.npz', ramsz=0.35)

    def tearDown(self):
        self.tmp.cleanup()

    def test_report(self):
        lines = tomo_spec.vertical_offset_report(self.log, 3, 5)
        text = '\n'.join(lines)
        self.assertIn('tomography scan 3: ramsz 0.35 mm', text)
        self.assertIn('NF scan 5: ramsz -0.025 mm', text)
        self.assertIn('vertical_motor_position = ramsz_nf - ramsz_tomo = -0.375 mm', text)

    def test_report_with_a_mask_lists_the_layers_the_nf_run_would_use(self):
        text = '\n'.join(tomo_spec.vertical_offset_report(self.log, 3, 4, self.mask, (-0.06, 0.06)))
        self.assertIn('uses 1 mask layer(s)', text)      # NF scan 4 has the same ramsz as the mask: layer at v = 0
        self.assertIn('layer ids [3]', text)

    def test_report_says_when_the_mask_does_not_cover_the_nf_layer(self):
        text = '\n'.join(tomo_spec.vertical_offset_report(self.log, 3, 5, self.mask, (-0.025, 0.025)))
        self.assertIn('matches NO mask layer', text)

    def test_report_warns_if_the_mask_is_from_another_scan(self):
        text = '\n'.join(tomo_spec.vertical_offset_report(self.log, 2, 5, self.mask, (-0.5, 0.5)))
        self.assertIn('WARNING: the mask records ramsz 0.35 but scan 2 is at -0.35', text)

    def run_command(self, *argv):
        import argparse
        parser = argparse.ArgumentParser()
        tomo_cli.configure_parser(parser.add_subparsers(dest='cmd'))
        args = parser.parse_args(['tomo', 'offset', *argv])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                args.func(args, parser)
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_command_prints_the_offset(self):
        code, out, _ = self.run_command(str(self.log), '3', '5')
        self.assertEqual(code, 0)
        self.assertIn('-0.375 mm', out)

    def test_command_with_mask_and_span(self):
        code, out, _ = self.run_command(str(self.log), '3', '4', '--mask', str(self.mask), '--span', '-0.06', '0.06')
        self.assertEqual(code, 0)
        self.assertIn('layer ids [3]', out)

    def test_command_exits_1_without_a_position(self):
        bare = self.dir / 'bare.log'
        bare.write_text(NO_RAMSZ)
        code, _, err = self.run_command(str(bare), '1', '1')
        self.assertEqual(code, 1)
        self.assertIn('no vertical position', err)

    def test_mask_and_span_go_together(self):
        code, _, _ = self.run_command(str(self.log), '3', '4', '--mask', str(self.mask))
        self.assertEqual(code, 2)


class TestConfigKeys(unittest.TestCase):

    def test_nf_config_reads_spec_log_and_nf_scan_and_defaults_to_none(self):
        base = {'NF_reconstruction': {'tomography': {'mask_filepath': 'm.npz'}}}
        tomo = nf_config.nf_root.NFRootConfig(base).reconstruction.tomography
        self.assertEqual((tomo['spec_log'], tomo['nf_scan'], tomo['vertical_motor_position']), (None, None, None))
        base['NF_reconstruction']['tomography'].update(spec_log='spec.log', nf_scan=29, vertical_motor_position=-1.0)
        tomo = nf_config.nf_root.NFRootConfig(base).reconstruction.tomography
        self.assertEqual((tomo['spec_log'], tomo['nf_scan'], tomo['vertical_motor_position']), ('spec.log', 29, -1.0))

    def test_tomography_config_vertical_motor_position_defaults_to_none(self):
        cfg = nf_config.nf_root.NFRootConfig({'tomography': {}})
        self.assertIsNone(cfg.tomography.reconstruction.vertical_motor_position)


if __name__ == '__main__':
    unittest.main()
