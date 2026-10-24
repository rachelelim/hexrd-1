"""Tests for ff_hedm/ff_grains_io.py, the file handling shared by the FF stitching and recompile scripts."""
import contextlib
import io as stdio
import os
import sys
import tempfile
import unittest

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ff_hedm'))

import ff_grains_io as io  # noqa: E402


def make_grains(y, comp=1.0, chi2=0.001):
    """grains.out style array of grains at the heights y, with strains that are 1e-3"""
    n = len(y)
    data = np.zeros([n, 21])
    data[:, 0] = np.arange(n)
    data[:, 1] = comp
    data[:, 2] = chi2
    data[:, 7] = y
    data[:, 15:18] = 1e-3
    return data


class TestFileNames(unittest.TestCase):

    def test_names(self):
        self.assertEqual(io.master_file_name('s_', 0), 's_S000_master.out')
        self.assertEqual(io.master_file_name('s_', 12), 's_S012_master.out')
        self.assertEqual(io.analysis_name('s_', 3, 1), 's_S3_L1')
        self.assertEqual(io.grain_nums_file_name(2, 0), 'grain_nums_S2_L0.txt')
        self.assertEqual(io.yaml_file_name('s_', 4), 's_S4.yml')
        self.assertEqual(io.beam_calibration_file_name('s_'), 's_beam_calibration.yml')

    def test_seed_names_by_location(self):
        self.assertEqual(io.seed_file_name(1, 2), 'S1_seed_L2_grains.out')
        self.assertEqual(io.seed_file_name(1, 2, 'above'), 'S1_seed_above_L2_grains.out')
        self.assertEqual(io.seed_file_name(1, 2, 'below'), 'S1_seed_below_L2_grains.out')


class TestLayerBounds(unittest.TestCase):
    # three layers, the highest first, as with offsets of -ramsz
    off = np.array([1.0, 0.0, -1.0])

    def test_both(self):
        b = io.layer_bounds(self.off, 0.4, 'both', beam_center_y=0.1, seed_margin=0.5)
        # illuminated volume 0.4 high around the beam center, and 0.5 beam heights beyond it
        self.assertAlmostEqual(b[1][0], 0.1 - 0.2 - 0.2)
        self.assertAlmostEqual(b[1][1], 0.1 + 0.2 + 0.2)

    def test_above_and_below_have_the_margin_on_one_side(self):
        above = io.layer_bounds(self.off, 0.4, 'above', beam_center_y=0.1, seed_margin=0.5)
        below = io.layer_bounds(self.off, 0.4, 'below', beam_center_y=0.1, seed_margin=0.5)
        self.assertAlmostEqual(above[1][0], 0.1 - 0.2)
        self.assertAlmostEqual(above[1][1], 0.1 + 0.2 + 0.2)
        self.assertAlmostEqual(below[1][0], 0.1 - 0.2 - 0.2)
        self.assertAlmostEqual(below[1][1], 0.1 + 0.2)

    def test_margin_is_a_fraction_of_the_beam_height(self):
        # the old windows reached 1.5 beam heights from the center of the layer
        b = io.layer_bounds(self.off, 0.35, 'both', seed_margin=1.0)
        self.assertAlmostEqual(b[1][0], -1.5 * 0.35)
        self.assertAlmostEqual(b[1][1], 1.5 * 0.35)
        none = io.layer_bounds(self.off, 0.35, 'both', seed_margin=0.0)
        self.assertAlmostEqual(none[1][1] - none[1][0], 0.35)

    def test_outer_layers_have_no_outer_limit(self):
        b = io.layer_bounds(self.off, 0.4, 'both')
        self.assertEqual(b[0][1], np.inf)
        self.assertEqual(b[2][0], -np.inf)
        self.assertTrue(np.isfinite(b[0][0]) and np.isfinite(b[2][1]))
        self.assertTrue(all(np.isfinite(b[1])))

    def test_outer_layers_do_not_depend_on_the_order_of_the_layers(self):
        b = io.layer_bounds(self.off[::-1], 0.4, 'above')
        self.assertEqual(b[2][1], np.inf)
        self.assertEqual(b[0][0], -np.inf)

    def test_vertical_position_correction_moves_the_windows(self):
        b0 = io.layer_bounds(self.off, 0.4, 'both')
        b1 = io.layer_bounds(self.off, 0.4, 'both', v_pos_corr=0.05)
        self.assertAlmostEqual(b0[1][0] - b1[1][0], 0.05)

    def test_unknown_location(self):
        with self.assertRaises(ValueError):
            io.layer_bounds(self.off, 0.4, 'top')


class TestWellPopulatedRange(unittest.TestCase):

    def test_trims_the_sparse_tails(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([rng.uniform(-0.2, 0.15, 3000),  # dense core
                            [-0.3, -0.27, 0.19, 0.22, 0.25]])  # a few stray grains
        lo, hi = io.well_populated_range(y, 0.025, 10)
        self.assertGreater(lo, -0.26)
        self.assertLess(hi, 0.19)
        self.assertLess(lo, -0.18)
        self.assertGreater(hi, 0.13)

    def test_stops_at_a_gap(self):
        y = np.concatenate([np.linspace(0.0, 0.1, 400), np.linspace(0.3, 0.4, 400)])
        lo, hi = io.well_populated_range(y, 0.025, 10)
        self.assertLessEqual(hi, 0.13)  # does not cross to the second cluster or include the gap


class TestEnergyGradient(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = rng.uniform(-0.2, 0.15, 1500)
        self.true = 4e-4 * np.sin(2 * np.pi * self.y / 0.25) + 3e-4
        self.strain = self.true + rng.normal(0, 5e-5, len(self.y))

    def test_spline_follows_a_curve_a_line_cannot(self):
        spline = io.fit_energy_gradient(self.y, self.strain, 'spline', n_knots=6)
        line = io.fit_energy_gradient(self.y, self.strain, 'polynomial', degree=1)
        err_spline = np.abs(io.energy_gradient_strain(spline, self.y) - self.true).max()
        err_line = np.abs(io.energy_gradient_strain(line, self.y) - self.true).max()
        self.assertLess(err_spline, 1.5e-4)
        self.assertGreater(err_line, 3e-4)

    def test_a_line_is_recovered_by_degree_one(self):
        y = np.linspace(-0.1, 0.1, 50)
        g = io.fit_energy_gradient(y, 2e-3 * y + 1e-4, 'polynomial', degree=1)
        np.testing.assert_allclose(g['coefficients'], [2e-3, 1e-4], atol=1e-12)

    def test_held_at_the_ends_outside_the_fit_range(self):
        g = io.fit_energy_gradient(self.y, self.strain, 'spline', n_knots=6)
        beyond = io.energy_gradient_strain(g, np.array([g['y_min'] - 5.0, g['y_max'] + 5.0]))
        edge = io.energy_gradient_strain(g, np.array([g['y_min'], g['y_max']]))
        np.testing.assert_allclose(beyond, edge)
        self.assertTrue(np.all(np.abs(beyond) < 2e-3))  # not extrapolated off to a large value

    def test_outliers_are_left_out(self):
        strain = self.strain.copy()
        strain[10] += 5e-3
        g = io.fit_energy_gradient(self.y, strain, 'spline', n_knots=4, outlier_sigma=4)
        self.assertFalse(g['used'][10])
        self.assertGreater(g['used'].mean(), 0.99)

    def test_save_and_load(self):
        g = io.fit_energy_gradient(self.y, self.strain, 'spline', n_knots=4)
        g['beam_center_y'] = -0.035
        with tempfile.TemporaryDirectory() as d:
            name = os.path.join(d, io.beam_calibration_file_name('s_'))
            io.save_beam_calibration(name, g)
            loaded = io.load_beam_calibration(name)
        self.assertNotIn('used', loaded)
        self.assertEqual(loaded['beam_center_y'], -0.035)
        np.testing.assert_allclose(io.energy_gradient_strain(loaded, self.y),
                                   io.energy_gradient_strain(g, self.y))

    def test_missing_file_says_what_to_do(self):
        with self.assertRaisesRegex(FileNotFoundError, 'calc_lparm'):
            io.load_beam_calibration('/no/such/file.yml')

    def test_correction_only_changes_the_normal_strains(self):
        g = io.fit_energy_gradient(self.y, self.strain, 'spline', n_knots=4)
        data = make_grains(self.y[:20])
        data[:, 3:15] = 0.5
        data[:, 18:21] = 2e-4
        fixed = io.correct_energy_gradient(data, g)
        shift = io.energy_gradient_strain(g, data[:, 7])
        for col in (15, 16, 17):
            np.testing.assert_allclose(fixed[:, col], data[:, col] - shift)
        untouched = [c for c in range(21) if c not in (15, 16, 17)]
        np.testing.assert_array_equal(fixed[:, untouched], data[:, untouched])
        self.assertTrue(np.all(data[:, 15:18] == 1e-3))  # the input is not changed

    def test_corrected_strain_is_flat(self):
        g = io.fit_energy_gradient(self.y, self.strain, 'spline', n_knots=6)
        data = make_grains(self.y)
        data[:, 15:18] = self.strain[:, None]
        fixed = io.correct_energy_gradient(data, g)
        self.assertLess(fixed[:, 15:18].mean(1).std(), 0.3 * self.strain.std())


class TestGrainLists(unittest.TestCase):
    pos_offset = np.zeros([3, 3])
    pos_offset[1] = [0.4, 0.0, -0.4]  # the highest layer first

    def setUp(self):
        # grains from below the lowest layer to above the highest one
        self.y = np.linspace(-1.0, 1.0, 81)
        self.data = make_grains(self.y)
        self.data[::3, 1] = 0.3  # every third grain is a poor fit
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def lists(self, **kw):
        with contextlib.redirect_stdout(stdio.StringIO()):
            io.make_grain_lists('s_', self.data.copy(), 0, 0.35, self.pos_offset, self.tmp.name, **kw)
        return [np.loadtxt(os.path.join(self.tmp.name, io.grain_nums_file_name(0, i))).astype(int)
                for i in range(3)]

    def test_every_grain_is_in_a_list(self):
        for loc in io.LOCATIONS:
            lists = self.lists(location=loc)
            self.assertEqual(len(np.unique(np.concatenate(lists))), len(self.y), loc)

    def test_lists_are_the_windows(self):
        lists = self.lists(location='both', seed_margin=0.5, beam_center_y=-0.05)
        # the middle layer: the illuminated 0.35 around -0.05 and 0.175 beyond it
        inside = np.where((self.y >= -0.05 - 0.175 - 0.175) & (self.y <= -0.05 + 0.175 + 0.175))[0]
        np.testing.assert_array_equal(lists[1], inside)

    def test_seed_files_are_in_the_frame_of_the_layer(self):
        lists = self.lists(location='both')
        for i in range(3):
            seed = np.loadtxt(os.path.join(self.tmp.name, io.seed_file_name(1, i)))
            np.testing.assert_allclose(seed[:, 7], self.y[lists[i]] - self.pos_offset[1, i])
            np.testing.assert_array_equal(seed[:, 0], np.arange(len(lists[i])))  # renumbered

    def test_thresholds_leave_grains_out(self):
        everything = self.lists(location='both')
        good = self.lists(location='both', low_comp_thresh=0.6, high_chi2_thresh=0.01)
        for a, b in zip(everything, good):
            self.assertLess(len(b), len(a))
            self.assertTrue(np.all(self.data[b, 1] >= 0.6))

    def test_master_file_round_trips(self):
        with contextlib.redirect_stdout(stdio.StringIO()):
            io.write_grains_files(self.tmp.name, 's_', self.data, 7)
        back = np.loadtxt(os.path.join(self.tmp.name, io.master_file_name('s_', 7)))
        np.testing.assert_allclose(back, self.data, rtol=1e-11, atol=1e-12)


class TestYaml(unittest.TestCase):

    def test_config_for_the_next_state(self):
        with tempfile.TemporaryDirectory() as d:
            init = os.path.join(d, 'init.yml')
            with open(init, 'w') as f:
                yaml.dump({'analysis_name': 'old', 'image_series': {'format': 'frame-cache', 'data': []},
                           'fit_grains': {'estimate': None, 'npdiv': 2}}, f)
            panels = {'ff1': ['ff1_0_0', 'ff1_0_1'], 'ff2': ['ff2_0_0']}
            io.write_yaml(init, 'img/%s_%d_%s_%06d.npz', 'samp', 's_', 3, 3, 88, 403, d, panels, 'above')
            with open(os.path.join(d, io.yaml_file_name('s_', 3))) as f:
                docs = list(yaml.safe_load_all(f))
        self.assertEqual(len(docs), 3)  # the full config, then an update for each other layer
        self.assertEqual([x['analysis_name'] for x in docs], ['s_S3_L0', 's_S3_L1', 's_S3_L2'])
        self.assertEqual(docs[0]['fit_grains']['npdiv'], 2)  # the first has the rest of the config
        self.assertNotIn('npdiv', docs[1]['fit_grains'])
        self.assertEqual([x['fit_grains']['estimate'] for x in docs],
                         [io.seed_file_name(3, i, 'above') for i in range(3)])
        files = [[e['file'] for e in x['image_series']['data']] for x in docs]
        self.assertEqual(files[1], ['img/samp_89_ff1_000404.npz', 'img/samp_89_ff2_000404.npz'])
        self.assertEqual([e['panel'] for e in docs[2]['image_series']['data']],
                         [['ff1_0_0', 'ff1_0_1'], ['ff2_0_0']])


class TestInstrumentAndMaterialFiles(unittest.TestCase):

    def test_detector_panels_are_grouped(self):
        with tempfile.TemporaryDirectory() as d:
            name = os.path.join(d, 'inst.hexrd')
            with h5py.File(name, 'w') as f:
                dets = f.create_group('instrument/detectors')
                for panel, group in (('ff1_0_0', 'ff1'), ('ff1_0_1', 'ff1'), ('ff2_0_0', 'ff2')):
                    dets.create_group(panel)['group'] = group.encode()
                dets.create_group('lone')  # a panel that is not in a group
                f['instrument/beam/energy'] = np.array([61.332])
            panels = io.read_detector_panels(name)
            energy = io.read_beam_energy(name)
        self.assertEqual(panels, {'ff1': ['ff1_0_0', 'ff1_0_1'], 'ff2': ['ff2_0_0'], 'lone': ['lone']})
        self.assertAlmostEqual(energy, 61.332)

    def test_lattice_parameter_is_in_angstrom(self):
        with tempfile.TemporaryDirectory() as d:
            name = os.path.join(d, 'mats.h5')
            with h5py.File(name, 'w') as f:
                f['316L/LatticeParameters'] = np.array([0.35916, 0.35916, 0.35916, 90., 90., 90.])
            self.assertAlmostEqual(io.read_lattice_parameter(name, '316L'), 3.5916)


if __name__ == '__main__':
    unittest.main()
