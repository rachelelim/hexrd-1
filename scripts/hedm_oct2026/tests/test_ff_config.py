"""Tests for ff_hedm/ff_config.py, the settings file of the FF pipeline."""
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FF_DIR = os.path.join(HERE, '..', 'ff_hedm')
sys.path.insert(0, FF_DIR)

import ff_config  # noqa: E402

MINIMAL = {
    'sample': 'samp',
    'ff_dir': 'FF',
    'detector_file': 'det.hexrd',
    'materials': {'file': 'mats.h5', 'name': 'steel'},
    'layers': {'ramsz': [-0.4, 0.0, 0.4], 'beam_height': 0.3},
}


class ConfigFile:
    """A config in a temporary directory"""

    def __init__(self, test, settings):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.dir = os.path.realpath(self.tmp.name)
        self.path = os.path.join(self.dir, 'cfg.yml')
        with open(self.path, 'w') as f:
            yaml.dump(settings, f)

    def load(self):
        return ff_config.load_config(self.path)


class TestLoadConfig(unittest.TestCase):

    def test_defaults_and_derived_values(self):
        c = ConfigFile(self, MINIMAL).load()
        self.assertEqual(c.sample, 'samp')
        self.assertEqual(c.analysis_stem, 'samp_')
        self.assertEqual(c.num_layers, 3)
        self.assertEqual(c.beam_height, 0.3)
        self.assertEqual((c.location, c.seed_margin), ('both', 0.5))
        self.assertEqual(c.stitch.low_comp_thresh, 0.5)
        self.assertEqual(c.recompile.close_completeness_cutoff, 0.1)
        self.assertEqual(c.calibration.model, 'spline')
        self.assertEqual(c.hexrd.command, 'hexrd fit-grains')

    def test_layer_offset_is_minus_ramsz(self):
        c = ConfigFile(self, MINIMAL).load()
        self.assertEqual(c.pos_offset.shape, (3, 3))
        np.testing.assert_allclose(c.pos_offset[1], [0.4, 0.0, -0.4])
        self.assertTrue(np.all(c.pos_offset[[0, 2]] == 0))

    def test_paths_are_relative_to_the_config(self):
        f = ConfigFile(self, {**MINIMAL, 'output_dir': 'out/here'})
        c = f.load()
        self.assertEqual(c.ff_dir, os.path.join(f.dir, 'FF'))
        self.assertEqual(c.detector_file, os.path.join(f.dir, 'det.hexrd'))
        self.assertEqual(c.materials_file, os.path.join(f.dir, 'mats.h5'))
        self.assertEqual(c.output_dir, os.path.join(f.dir, 'out', 'here'))
        # the calibration is made into the output directory
        self.assertEqual(c.beam_calibration_file, os.path.join(f.dir, 'out', 'here', 'samp_beam_calibration.yml'))

    def test_absolute_paths_are_kept_and_output_defaults_to_the_config_directory(self):
        f = ConfigFile(self, {**MINIMAL, 'ff_dir': '/data/FF'})
        c = f.load()
        self.assertEqual(c.ff_dir, '/data/FF')
        self.assertEqual(c.output_dir, f.dir)

    def test_calibration_can_be_turned_off(self):
        c = ConfigFile(self, {**MINIMAL, 'calibration': {'apply': False}}).load()
        self.assertIsNone(c.beam_calibration_file)

    def test_settings_of_a_section_are_merged_with_the_defaults(self):
        c = ConfigFile(self, {**MINIMAL, 'stitch': {'dist': 0.5}}).load()
        self.assertEqual(c.stitch.dist, 0.5)
        self.assertEqual(c.stitch.misorientation, 0.25)

    def test_states(self):
        c = ConfigFile(self, {**MINIMAL, 'states': {1: {'first_scan': 5, 'first_file': 100},
                                                    2: {'first_scan': 9, 'first_file': 104}}}).load()
        self.assertEqual(c.states, {1: (5, 100), 2: (9, 104)})

    def test_the_example_config_loads(self):
        c = ff_config.load_config(os.path.join(FF_DIR, 'ff_pipeline.yml'))
        self.assertEqual(c.num_layers, len(c.ramsz))
        self.assertEqual(sorted(c.states), list(range(1, len(c.states) + 1)))


class TestBadConfigs(unittest.TestCase):

    def check(self, settings, text):
        with self.assertRaisesRegex(ff_config.ConfigError, text):
            ConfigFile(self, settings).load()

    def test_a_misspelt_setting(self):
        self.check({**MINIMAL, 'stitch': {'dist_thresh': 0.1}}, 'dist_thresh')
        self.check({**MINIMAL, 'smaple': 'x'}, 'smaple')

    def test_missing_settings(self):
        for key in ('sample', 'ff_dir', 'detector_file'):
            self.check({k: v for k, v in MINIMAL.items() if k != key}, key)
        self.check({**MINIMAL, 'layers': {'ramsz': [0.0]}}, 'beam_height')
        self.check({**MINIMAL, 'materials': {'file': 'm.h5'}}, 'name')

    def test_bad_values(self):
        self.check({**MINIMAL, 'seeds': {'location': 'top'}}, 'location')
        self.check({**MINIMAL, 'seeds': {'margin': -1}}, 'negative')
        self.check({**MINIMAL, 'calibration': {'model': 'cubic'}}, 'model')

    def test_states_have_to_be_numbered_from_one_with_none_missing(self):
        scans = {'first_scan': 1, 'first_file': 1}
        self.check({**MINIMAL, 'states': {2: scans}}, 'numbered 1, 2, 3')
        self.check({**MINIMAL, 'states': {1: scans, 3: scans}}, 'numbered 1, 2, 3')
        self.check({**MINIMAL, 'states': {1: {'first_scan': 1}}}, 'first_file')

    def test_what_the_later_states_need(self):
        c = ConfigFile(self, MINIMAL).load()
        with self.assertRaisesRegex(ff_config.ConfigError, 'initial_yaml.*image_fstem.*states'):
            ff_config.require_for_hexrd(c)


if __name__ == '__main__':
    unittest.main()
