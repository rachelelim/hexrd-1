"""End to end test of the FF pipeline on synthetic grains: calc_lparm, stitch_grains, build_pipeline and
recompile_grains, one after the other, with a simulated hexrd fit between the first state and the next."""
import contextlib
import io as stdio
import os
import stat
import sys
import tempfile
import unittest

import h5py
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ff_hedm'))

import build_pipeline  # noqa: E402
import calc_lparm  # noqa: E402
import ff_config  # noqa: E402
import ff_grains_io as io  # noqa: E402
import recompile_grains  # noqa: E402
import stitch_grains  # noqa: E402

MATERIALS = os.path.join(HERE, '..', '..', '..', 'tests', 'data', 'testmat.h5')  # xtal1 is cubic

RAMSZ = [-0.1, 0.1]  # two layers, 0.2 mm apart, so that their 0.35 mm beams overlap
BEAM = 0.35
BEAM_CENTER = -0.03
N_GRAINS = 1500


def true_gradient(y):
    """The strain error of the beam that the synthetic grains are made with"""
    return 8e-4 * np.sin((y + 0.05) / 0.1)


def quiet():
    return contextlib.redirect_stdout(stdio.StringIO())


def make_grains(rng):
    """Physical grains, positions in the frame of the sample, which the layers see parts of"""
    n = N_GRAINS
    grains = np.zeros([n, 21])
    grains[:, 1] = 0.97  # completeness
    grains[:, 2] = 0.003  # chi2
    grains[:, 3:6] = rng.uniform(-0.6, 0.6, [n, 3])  # orientations
    grains[:, 6] = rng.uniform(-0.3, 0.3, n)
    grains[:, 7] = rng.uniform(-0.3, 0.25, n)  # sample frame y
    grains[:, 8] = rng.uniform(-0.3, 0.3, n)
    grains[:, 9:12] = 1.0  # inverse of the stretch, no strain
    return grains


def layer_grains(grains, layer, rng):
    """grains.out of a layer, as hexrd gives it: what is in the beam, in the frame of the layer"""
    offset = -RAMSZ[layer]
    y = grains[:, 7] - offset
    seen = (y >= BEAM_CENTER - BEAM / 2) & (y <= BEAM_CENTER + BEAM / 2)
    data = grains[seen].copy()
    data[:, 7] = y[seen]
    data[:, 6:9] += rng.normal(0, 1e-5, [seen.sum(), 3])
    # the strain of the grains is a little of their own and the gradient of the beam
    data[:, 15:18] = rng.normal(0, 2e-5, [seen.sum(), 1]) + true_gradient(y[seen])[:, None]
    data[:, 0] = np.arange(len(data))
    return data


def write_grains_out(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data)


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = os.path.realpath(cls.tmp.name)
        cls.dir = d
        rng = np.random.default_rng(3)
        cls.grains = make_grains(rng)
        cls.layer_data = [layer_grains(cls.grains, i, rng) for i in range(2)]
        for i, data in enumerate(cls.layer_data):
            write_grains_out(os.path.join(d, 'FF', 'samp_S0_L%d' % i, 'grains.out'), data)

        with h5py.File(os.path.join(d, 'det.hexrd'), 'w') as f:
            dets = f.create_group('instrument/detectors')
            for panel in ('ff1_0_0', 'ff1_0_1', 'ff2_0_0'):
                dets.create_group(panel)['group'] = panel.split('_')[0].encode()
            f['instrument/beam/energy'] = np.array([61.332])
        with open(os.path.join(d, 'init.yml'), 'w') as f:
            yaml.dump({'analysis_name': 'samp_S0', 'image_series': {'format': 'frame-cache', 'data': []},
                       'fit_grains': {'estimate': None, 'npdiv': 2}}, f)

        settings = {
            'sample': 'samp', 'ff_dir': 'FF', 'output_dir': 'out', 'detector_file': 'det.hexrd',
            'materials': {'file': os.path.realpath(MATERIALS), 'name': 'xtal1'},
            'layers': {'ramsz': RAMSZ, 'beam_height': BEAM},
            # 0.002 mm apart is the same grain, the synthetic grains are never that close otherwise
            'stitch': {'dist': 0.002},
            'calibration': {'min_grains_per_bin': 5, 'n_knots': 6},
            'hexrd': {'initial_yaml': 'init.yml', 'image_fstem': 'img/%s_%d_%s_%06d.npz'},
            'states': {1: {'first_scan': 20, 'first_file': 200}, 2: {'first_scan': 24, 'first_file': 204}},
        }
        with open(os.path.join(d, 'cfg.yml'), 'w') as f:
            yaml.dump(settings, f)
        cls.cfg = ff_config.load_config(os.path.join(d, 'cfg.yml'))

        with quiet():
            cls.calibration = calc_lparm.calibrate(cls.cfg)
            cls.master = stitch_grains.stitch(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out(self, name):
        return os.path.join(self.cfg.output_dir, name)

    def test_calibration_finds_the_gradient_and_the_beam(self):
        g = self.calibration
        ys = np.linspace(-0.12, 0.08, 20)
        err = io.energy_gradient_strain(g, ys) - true_gradient(ys)
        err -= err.mean()  # the mean strain is not known, it is taken as zero
        self.assertLess(np.abs(err).max(), 1e-4)
        self.assertAlmostEqual(g['beam_center_y'], BEAM_CENTER, delta=0.03)
        for name in ('samp_beam_calibration.yml', 'samp_energy_gradient.png', 'samp_grain_density.png'):
            self.assertTrue(os.path.exists(self.out(name)), name)

    def test_stitch_merges_the_grains_seen_in_both_layers(self):
        seen = sum(len(d) for d in self.layer_data)
        self.assertGreater(seen, len(self.master))  # some are in both layers
        # each physical grain is one grain of the master list
        in_beam = np.zeros(len(self.grains), bool)
        for i in range(2):
            y = self.grains[:, 7] + RAMSZ[i]
            in_beam |= (y >= BEAM_CENTER - BEAM / 2) & (y <= BEAM_CENTER + BEAM / 2)
        self.assertEqual(len(self.master), in_beam.sum())
        np.testing.assert_array_equal(self.master[:, 0], np.arange(len(self.master)))

    def test_stitch_takes_out_the_gradient_and_moves_the_layers_to_the_sample(self):
        # in the frame of the layer the strain follows the gradient, after the stitch it is flat in y
        raw = np.concatenate([d[:, 15:18].mean(1) for d in self.layer_data])
        self.assertGreater(raw.std(), 3 * self.master[:, 15:18].mean(1).std())
        # a grain keeps the position in the sample that it was made at
        pos = self.grains[:, 6:9]
        nearest = np.array([np.abs(pos - p).sum(1).min() for p in self.master[:, 6:9]])
        self.assertLess(nearest.max(), 1e-3)

    def test_files_for_the_next_state(self):
        for name in ('samp_S000_master.out', 'S1_seed_L0_grains.out', 'S1_seed_L1_grains.out',
                     'grain_nums_S0_L0.txt', 'grain_nums_S0_L1.txt'):
            self.assertTrue(os.path.exists(self.out(name)), name)
        lists = [np.loadtxt(self.out('grain_nums_S0_L%d.txt' % i)).astype(int) for i in range(2)]
        self.assertEqual(len(np.unique(np.concatenate(lists))), len(self.master))  # none left out

    def test_build_pipeline_writes_the_configs_and_scripts(self):
        written = build_pipeline.build_pipeline(self.cfg)
        names = sorted(os.path.basename(p) for p in written)
        self.assertEqual(names, ['run_pipeline.sh', 'run_state_1.sh', 'run_state_2.sh',
                                 'samp_S1.yml', 'samp_S2.yml'])
        with open(self.out('samp_S2.yml')) as f:
            docs = list(yaml.safe_load_all(f))
        self.assertEqual([x['analysis_name'] for x in docs], ['samp_S2_L0', 'samp_S2_L1'])
        self.assertEqual(docs[1]['fit_grains']['estimate'], 'S2_seed_L1_grains.out')
        self.assertEqual([e['file'] for e in docs[1]['image_series']['data']],
                         ['img/samp_25_ff1_000205.npz', 'img/samp_25_ff2_000205.npz'])
        self.assertEqual([e['panel'] for e in docs[0]['image_series']['data']],
                         [['ff1_0_0', 'ff1_0_1'], ['ff2_0_0']])
        with open(self.out('run_state_2.sh')) as f:
            script = f.read()
        self.assertIn('hexrd fit-grains samp_S2.yml', script)
        self.assertIn('recompile_grains.py', script)
        self.assertTrue(script.rstrip().endswith('--state 2'))
        self.assertTrue(os.stat(self.out('run_state_2.sh')).st_mode & stat.S_IXUSR)
        with open(self.out('run_pipeline.sh')) as f:
            lines = f.read().splitlines()
        order = [next(i for i, line in enumerate(lines) if text in line)
                 for text in ('calc_lparm.py', 'stitch_grains.py', 'run_state_1.sh', 'run_state_2.sh')]
        self.assertEqual(order, sorted(order))

    def test_build_one_state(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = ff_config.load_config(self.cfg.path)
            cfg.output_dir = d
            written = build_pipeline.build_pipeline(cfg, only_state=1)
            self.assertEqual(sorted(os.path.basename(p) for p in written), ['run_state_1.sh', 'samp_S1.yml'])
        with self.assertRaisesRegex(ff_config.ConfigError, 'State 7'):
            build_pipeline.build_pipeline(self.cfg, only_state=7)

    def test_recompile_reproduces_the_master_from_unchanged_fits(self):
        """hexrd, started from the seeds and finding the same grains, gives back the master list"""
        cfg = ff_config.load_config(self.cfg.path)
        with tempfile.TemporaryDirectory() as d:
            for name in os.listdir(self.cfg.output_dir):
                if name.startswith(('S1_seed', 'grain_nums', 'samp_S000', 'samp_beam')):
                    with open(os.path.join(self.cfg.output_dir, name), 'rb') as src:
                        with open(os.path.join(d, name), 'wb') as dst:
                            dst.write(src.read())
            cfg.output_dir = d
            cfg.beam_calibration_file = os.path.join(d, 'samp_beam_calibration.yml')
            calibration = io.load_beam_calibration(cfg.beam_calibration_file)
            for i in range(2):
                seed = np.loadtxt(os.path.join(d, io.seed_file_name(1, i)), ndmin=2)
                seed[:, 15:18] += io.energy_gradient_strain(calibration, seed[:, 7])[:, None]  # as hexrd sees it
                write_grains_out(io.grains_file_path(d, 'samp_', 1, i), seed)
            with quiet():
                recompiled = recompile_grains.recompile(cfg, 1)
            np.testing.assert_allclose(recompiled, self.master, rtol=0, atol=1e-9)
            for name in ('samp_S001_master.out', 'S2_seed_L0_grains.out', 'grain_nums_S1_L1.txt'):
                self.assertTrue(os.path.exists(os.path.join(d, name)), name)

    def test_recompile_is_only_for_later_states(self):
        with self.assertRaises(ValueError):
            recompile_grains.recompile(self.cfg, 0)


if __name__ == '__main__':
    unittest.main()
