"""``hexrd tomo pipeline`` on a synthetic scan: spec log with a dark, a bright and a tomography scan.

The qc stage reads a band of detector rows that only a real detector has, so it is not run here.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

import tomo_testing as T
from hexrd.hedm import nf_config
from hexrd.hedm.grainmap.tomo_options import PipelineOptions
from hexrd.hedm.grainmap.tomo_pipeline import PipelineError, run_pipeline


def iou(mask, ideal):
    return (mask & ideal).sum() / (mask | ideal).sum()


class Run:
    """A synthetic scan, its spec log and a pipeline yml in a temporary folder."""

    def __init__(self, dropped=(), v_bnds=None, swap=False, **yml):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = self.dir / 'data'
        self.data.mkdir()
        dark, bright, tomo = T.write_scan_files(self.data, 90, dropped)
        if swap:
            dark.rename(self.data / 'tmp.h5')
            bright.rename(dark)
            (self.data / 'tmp.h5').rename(bright)
        self.log = T.write_spec_log(self.dir / 'spec.log', 90, dropped, tomo_h5_dir='/not/here')
        self.detector = T.write_instrument(self.dir / 'det.yml')
        self.yml_path = T.write_config(self.dir / 'pipeline.yml', self.detector, spec_log=self.log, spec_scan=3,
                                       tomo_dir=self.data, analysis='synthetic', v_bnds=v_bnds, **yml)
        self.run_dir = self.data / 'run_scan3'

    def cfg(self):
        return nf_config.open_file(str(self.yml_path))[0]

    def cleanup(self):
        self.tmp.cleanup()


STAGES = ['setup', 'missing', 'recon', 'smooth']        # no qc: see the module docstring


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sr = Run(dropped=(10, 40, 41))
        cls.final = Path(run_pipeline(cls.sr.cfg(), PipelineOptions(stages=STAGES, smooth_sigma=1.0, powder_open_um=0)))

    @classmethod
    def tearDownClass(cls):
        cls.sr.cleanup()

    def test_files_of_each_stage(self):
        names = sorted(p.name for p in self.sr.run_dir.iterdir())
        for expected in ('tomo_config.yml', 'retiga_000003_missing_slots.txt', 'synthetic_4.5um_tomo_mask.npz',
                         'synthetic_4.5um_tomo_mask_smooth.npz', 'recon.log', 'find_missing.log'):
            self.assertIn(expected, names)
        self.assertFalse([n for n in names if n.endswith('.dat')])

    def test_final_mask_is_the_smoothed_one(self):
        self.assertEqual(self.final.name, 'synthetic_4.5um_tomo_mask_smooth.npz')
        d = np.load(self.final)
        self.assertIn('mask_unsmoothed', d.files)
        self.assertEqual(d['mask'].shape[0], 4)

    def test_mask_is_the_disk_even_with_dropped_frames(self):
        mask = np.load(self.sr.run_dir / 'synthetic_4.5um_tomo_mask.npz')['mask']
        ideal = T.ideal_mask(mask.shape[1:])
        for layer in mask:
            self.assertGreater(iou(layer, ideal), 0.7)

    def test_default_powder_removal_deletes_a_disk_this_small(self):
        # powder_open_um defaults to 40 um, wider than this 18 um disk; the option works on the real scale
        run = Run()
        try:
            final = run_pipeline(run.cfg(), PipelineOptions(stages=['setup', 'missing', 'recon']))
            self.assertFalse(np.load(final)['mask'].any())
        finally:
            run.cleanup()

    def test_missing_slots_are_the_dropped_frames(self):
        slots = (self.sr.run_dir / 'retiga_000003_missing_slots.txt').read_text().split()
        self.assertEqual(slots, ['10', '40', '41'])

    def test_run_config_records_what_the_pipeline_found(self):
        d = yaml.safe_load((self.sr.run_dir / 'tomo_config.yml').read_text())
        t = d['tomography']
        self.assertEqual(Path(t['dark']['filename']).name, 'retiga_000001.h5')
        self.assertEqual(Path(t['bright']['filename']).name, 'retiga_000002.h5')
        self.assertEqual(Path(t['tomo_images']['filename']).name, 'retiga_000003.h5')
        for key in ('dark', 'bright'):
            self.assertTrue(Path(t[key]['filename']).is_absolute())
        self.assertEqual(t['tomo_images']['num_images'], 90)
        self.assertEqual(t['spec']['scan'], 3)
        self.assertEqual(d['analysis_name'], 'synthetic')
        self.assertEqual(Path(d['output_directory']).resolve(), self.sr.run_dir.resolve())
        self.assertEqual(t['reconstruction']['v_bnds'], T.V_BNDS)
        self.assertEqual(t['processing']['recon_thresh'], 0.0003)       # the user's value, not the default
        self.assertEqual(t['processing']['noise_obj_size'], 300)

    def test_run_config_can_be_read_back(self):
        cfg = nf_config.open_file(str(self.sr.run_dir / 'tomo_config.yml'))[0]
        self.assertEqual(cfg.tomography.tomo_images.num_imgs, 90)
        self.assertEqual(cfg.multiprocessing.num_cpus, 2)

    def test_existing_stages_are_skipped_and_forced_ones_are_redone(self):
        mask = self.sr.run_dir / 'synthetic_4.5um_tomo_mask.npz'
        before = mask.stat().st_mtime_ns
        with self.assertLogs('hexrd', level='INFO') as logs:
            run_pipeline(self.sr.cfg(), PipelineOptions(stages=STAGES, smooth_sigma=1.0, powder_open_um=0))
        text = '\n'.join(logs.output)
        self.assertIn('missing: using existing', text)
        self.assertIn('recon: using existing', text)
        self.assertIn('smooth: using existing', text)
        self.assertEqual(mask.stat().st_mtime_ns, before)
        run_pipeline(self.sr.cfg(), PipelineOptions(stages=['recon'], force=True, powder_open_um=0))
        self.assertGreater(mask.stat().st_mtime_ns, before)


class TestOptions(unittest.TestCase):

    def test_layers_give_a_named_mask_and_skip_qc_and_interp(self):
        run = Run()
        try:
            with self.assertLogs('hexrd', level='INFO') as logs:
                final = run_pipeline(run.cfg(), PipelineOptions(stages=['setup', 'missing', 'recon', 'qc', 'interp'],
                                                                layers=[1], interp='1-1'))
            self.assertEqual(Path(final).name, 'synthetic_layer1_4.5um_tomo_mask.npz')
            self.assertEqual(np.load(final)['mask'].shape[0], 1)
            self.assertIn('layers given: qc/interp need all layers, skipping them', '\n'.join(logs.output))
        finally:
            run.cleanup()

    def test_interpolating_a_layer_range(self):
        run = Run(v_bnds=[-0.0225, 0.0225])          # 10 layers
        try:
            final = Path(run_pipeline(run.cfg(), PipelineOptions(stages=['setup', 'missing', 'recon', 'interp'],
                                                                 interp='4-5')))
            self.assertEqual(final.name, 'synthetic_4.5um_tomo_mask_interp.npz')
            d = np.load(final)
            self.assertEqual(d['mask'].shape[0], 10)
            self.assertEqual(d['interpolated_layers'].tolist(), [4, 5])
            self.assertTrue(np.array_equal(d['mask'][4], d['mask'][3]))
        finally:
            run.cleanup()

    def test_dry_run_checks_the_inputs_and_writes_nothing_else(self):
        run = Run()
        try:
            final = run_pipeline(run.cfg(), PipelineOptions(dry_run=True))
            self.assertEqual([p.name for p in run.run_dir.iterdir()], [])
            self.assertFalse(Path(final).exists())
        finally:
            run.cleanup()

    def test_scan_flag_overrides_the_yml(self):
        run = Run()
        try:
            with self.assertRaisesRegex(PipelineError, 'scan 99 not in'):
                run_pipeline(run.cfg(), PipelineOptions(scan=99))
        finally:
            run.cleanup()

    def test_run_dir_option(self):
        run = Run()
        try:
            run_pipeline(run.cfg(), PipelineOptions(stages=['setup'], run_dir=str(run.dir / 'elsewhere')))
            self.assertTrue((run.dir / 'elsewhere' / 'tomo_config.yml').exists())
        finally:
            run.cleanup()


class TestSampleDir(unittest.TestCase):
    """The files are found as <sample_dir>/<scan>/nf/retiga_*.h5, not in tomo_dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.sample = self.dir / 'sample'
        self.sample.mkdir()
        self.dark, self.bright, self.tomo = T.write_sample_dir(self.sample)
        self.log = T.write_spec_log(self.dir / 'spec.log', 90, (), numbers=(21, 22, 23), tomo_h5_dir='/not/here')
        self.detector = T.write_instrument(self.dir / 'det.yml')

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, **more):
        kw = dict(spec_log=self.log, spec_scan=23, sample_dir=self.sample, analysis='synthetic')
        path = T.write_config(self.dir / 'pipeline.yml', self.detector, **{**kw, **more})
        return nf_config.open_file(str(path))[0]

    def setup_config(self, cfg):
        run_dir = self.dir / 'run'
        run_pipeline(cfg, PipelineOptions(stages=['setup'], run_dir=str(run_dir)))
        return yaml.safe_load((run_dir / 'tomo_config.yml').read_text())['tomography']

    def test_files_are_found_by_scan_number(self):
        t = self.setup_config(self.cfg())
        self.assertEqual(Path(t['dark']['filename']).resolve(), self.dark.resolve())
        self.assertEqual(Path(t['bright']['filename']).resolve(), self.bright.resolve())
        self.assertEqual(Path(t['tomo_images']['filename']).resolve(), self.tomo.resolve())
        self.assertEqual(t['spec']['sample_dir'], str(self.sample))

    def test_the_names_in_the_spec_log_are_not_needed(self):
        text = self.log.read_text().splitlines()
        self.log.write_text('\n'.join(line for line in text if not ('retiga_000021' in line or 'retiga_000022' in line)))
        t = self.setup_config(self.cfg())
        self.assertEqual(Path(t['dark']['filename']).resolve(), self.dark.resolve())

    def test_sample_dir_wins_over_tomo_dir(self):
        t = self.setup_config(self.cfg(tomo_dir=self.dir / 'nowhere'))
        self.assertEqual(Path(t['tomo_images']['filename']).resolve(), self.tomo.resolve())

    def test_spec_log_name_picks_one_of_several_files(self):
        spec_named = self.tomo.parent / 'retiga_000023.h5'      # the name in the spec log
        T.write_h5(spec_named, np.zeros((3, T.ROWS, T.COLS)))
        t = self.setup_config(self.cfg())
        self.assertEqual(Path(t['tomo_images']['filename']).name, 'retiga_000023.h5')

    def test_several_files_and_no_name_to_choose_by(self):
        T.write_h5(self.bright.parent / 'retiga_900999.h5', np.zeros((3, T.ROWS, T.COLS)))
        text = self.log.read_text().replace('retiga_000022', 'retiga_zzz')
        self.log.write_text(text)
        with self.assertRaisesRegex(PipelineError, 'bright: 2 retiga_\\*.h5 files'):
            run_pipeline(self.cfg(), PipelineOptions(stages=['setup']))

    def test_missing_scan_folder(self):
        import shutil
        shutil.rmtree(self.sample / '21')
        with self.assertRaisesRegex(PipelineError, 'dark: .*21/nf does not exist'):
            run_pipeline(self.cfg(), PipelineOptions(stages=['setup']))

    def test_folder_without_a_retiga_file(self):
        self.tomo.unlink()
        with self.assertRaisesRegex(PipelineError, 'tomo: no retiga_\\*.h5 file'):
            run_pipeline(self.cfg(), PipelineOptions(stages=['setup']))

    def test_file_set_in_the_yml_wins(self):
        other = self.dir / 'other_dark.h5'
        T.write_h5(other, np.full((5, T.ROWS, T.COLS), T.DARK))
        t = self.setup_config(self.cfg(extra_tomography={'dark': {'filename': str(other)}}))
        self.assertEqual(Path(t['dark']['filename']).name, 'other_dark.h5')
        self.assertEqual(Path(t['bright']['filename']).resolve(), self.bright.resolve())

    def test_tomo_file_set_in_the_yml_wins(self):
        other = self.dir / 'other_tomo.h5'
        T.write_h5(other, np.zeros((3, T.ROWS, T.COLS)))
        t = self.setup_config(self.cfg(extra_tomography={'tomo_images': {'filename': str(other)}}))
        self.assertEqual(Path(t['tomo_images']['filename']).name, 'other_tomo.h5')

    def test_reconstruction_from_a_sample_dir(self):
        cfg = self.cfg()
        final = run_pipeline(cfg, PipelineOptions(stages=['setup', 'missing', 'recon'], run_dir=str(self.dir / 'run'),
                                                  powder_open_um=0))
        mask = np.load(final)['mask']
        ideal = T.ideal_mask(mask.shape[1:])
        for layer in mask:
            self.assertGreater(iou(layer, ideal), 0.7)


class TestErrors(unittest.TestCase):

    def test_unknown_stage(self):
        run = Run()
        try:
            with self.assertRaisesRegex(PipelineError, 'unknown stage'):
                run_pipeline(run.cfg(), PipelineOptions(stages=['setup', 'polish']))
        finally:
            run.cleanup()

    def test_spec_log_and_scan_are_required(self):
        cfg = nf_config.nf_root.NFRootConfig({'input_files': {'detector_file': 'd.yml'}, 'tomography': {}})
        with self.assertRaisesRegex(PipelineError, 'spec log and scan'):
            run_pipeline(cfg, PipelineOptions())

    def test_swapped_dark_and_bright(self):
        run = Run(swap=True)
        try:
            with self.assertRaisesRegex(PipelineError, 'look swapped'):
                run_pipeline(run.cfg(), PipelineOptions(stages=['setup']))
        finally:
            run.cleanup()

    def test_missing_h5_files(self):
        run = Run()
        try:
            (run.data / 'retiga_000002.h5').unlink()
            with self.assertRaisesRegex(PipelineError, 'not found'):
                run_pipeline(run.cfg(), PipelineOptions(stages=['setup']))
        finally:
            run.cleanup()

    def test_a_dark_set_in_the_yml_replaces_the_one_from_the_spec_log(self):
        run = Run()
        try:
            other = run.dir / 'other_dark.h5'
            frames = np.full((5, T.ROWS, T.COLS), T.DARK)
            T.write_h5(other, frames)
            (run.data / 'retiga_000001.h5').unlink()        # the spec log's dark is gone
            cfg = run.cfg()
            cfg._cfg['tomography']['dark'] = {'filename': str(other)}
            run_pipeline(cfg, PipelineOptions(stages=['setup']))
            d = yaml.safe_load((run.run_dir / 'tomo_config.yml').read_text())
            self.assertEqual(Path(d['tomography']['dark']['filename']).name, 'other_dark.h5')
        finally:
            run.cleanup()

    def test_detector_file_is_required(self):
        run = Run()
        try:
            cfg = run.cfg()
            cfg._cfg['input_files']['detector_file'] = None
            with self.assertRaisesRegex(PipelineError, 'detector_file'):
                run_pipeline(cfg, PipelineOptions(stages=['setup']))
        finally:
            run.cleanup()


if __name__ == '__main__':
    unittest.main()
