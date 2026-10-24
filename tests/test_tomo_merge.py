"""Merging the tomography masks of one sample, scanned at several heights: hexrd tomo merge."""
import argparse
import logging
import os
import tempfile
import unittest

import numpy as np
import scipy.ndimage as ndi

from hexrd.hedm.cli import tomo as tomo_cli
from hexrd.hedm.grainmap import tomo_merge

VS = 0.01
N = 40  # voxels across the plane of a layer
STARTS = [0, 100, 200]  # first slice of each scan in the sample
LENGTHS = [120, 120, 80]
REF = 1


def make_sample():
    """A sample 280 slices tall, with features that change with height so that overlaps can be registered"""
    k = np.arange(280)[:, None, None]
    z, x = np.mgrid[0:N, 0:N]
    cz, cx = 19.5 + 3 * np.sin(k / 25.0), 19.5 + 3 * np.cos(k / 31.0)
    radius = 11 + 2.5 * np.sin(k / 17.0)
    body = (z - cz) ** 2 + (x - cx) ** 2 < radius ** 2
    hole = (z - 10) ** 2 + (x - 28) ** 2 < 3 ** 2
    block = (k % 90 < 30) & (np.abs(z - 31) < 3) & (np.abs(x - 12) < 5)
    return (body & ~hole) | block


def make_scan(truth, i, drift=(0, 0), extra_offset=0.0):
    """The mask of scan i, with the sample drifted by ``drift`` voxels, as a *_tomo_mask.npz would have it"""
    s, n = STARTS[i], LENGTHS[i]
    mask = truth[s:s + n].copy()
    if any(drift):
        mask = np.stack([ndi.shift(m.astype(float), drift, order=0, mode='nearest') > 0.5 for m in mask])
    y = (np.arange(n) - (n - 1) / 2) * VS  # heights in the frame of the scan
    gz, gx = np.mgrid[0:N, 0:N] * VS
    shape = (n, N, N)
    scan = {'mask': mask, 'voxel_spacing': np.array(VS),
            'Ys': np.broadcast_to(y[:, None, None], shape).copy(),
            'Xs': np.broadcast_to(gx, shape).copy(), 'Zs': np.broadcast_to(gz, shape).copy()}
    # how far the heights of the scan have to be shifted to be the heights in the sample
    sample_y0 = (s - STARTS[REF]) * VS + (y[0] if REF == i else 0.0)
    return scan, sample_y0, y[0]


def offsets_for(extra=(0.0, 0.0, 0.0)):
    """The offsets that put each scan at its place in the sample, in the frame of the reference scan"""
    refy0 = -(LENGTHS[REF] - 1) / 2 * VS
    return np.array([(refy0 + (STARTS[i] - STARTS[REF]) * VS) - (-(LENGTHS[i] - 1) / 2 * VS) + extra[i]
                     for i in range(3)])


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def truth_slices(merged_mask, truth):
    """The slices of the truth that the merged mask has, with the merged layer k being the sample slice
    start + k, where the start is that of the first scan"""
    return truth[STARTS[0]:STARTS[0] + len(merged_mask)]


class TestMerge(unittest.TestCase):

    def setUp(self):
        self.truth = make_sample()
        self.drifts = [(-2, 3), (0, 0), (2, -1)]  # how far each scan's sample has moved
        self.scans = [make_scan(self.truth, i, self.drifts[i])[0] for i in range(3)]
        self.offsets = offsets_for()

    def merge(self, **kw):
        return tomo_merge.merge_masks(self.scans, self.offsets, reference=REF, **kw)

    def test_the_sample_is_recovered_on_a_regular_grid(self):
        out = self.merge()
        self.assertEqual(out['mask'].shape, (280, N, N))
        y = out['Ys'][:, 0, 0]
        np.testing.assert_allclose(np.diff(y), VS, atol=1e-9)
        self.assertGreater(iou(out['mask'], self.truth), 0.97)
        # the volume is not grown by the shifts and the blending
        self.assertAlmostEqual(out['mask'].sum() / self.truth.sum(), 1.0, delta=0.01)

    def test_the_registration_finds_the_drift(self):
        out = self.merge()
        np.testing.assert_allclose(out['merge_shifts_voxels'][0], (2, -3), atol=0.3)
        np.testing.assert_allclose(out['merge_shifts_voxels'][1], (0, 0))
        np.testing.assert_allclose(out['merge_shifts_voxels'][2], (-2, 1), atol=0.3)
        # without it the drift stays in the merged mask
        unregistered = self.merge(register=False)
        self.assertLess(iou(unregistered['mask'], self.truth), iou(out['mask'], self.truth) - 0.03)

    def test_the_reference_is_kept_as_it_is_where_nothing_overlaps_it(self):
        out = self.merge()
        # layers of the reference that no other scan covers: slices 120..199 of the sample
        np.testing.assert_array_equal(out['mask'][120:200], self.scans[REF]['mask'][20:100])

    def test_coordinates_are_those_of_the_reference_and_the_merged_heights(self):
        out = self.merge()
        np.testing.assert_array_equal(out['Xs'][5], self.scans[REF]['Xs'][0])
        np.testing.assert_array_equal(out['Zs'][200], self.scans[REF]['Zs'][0])
        # the heights are on the grid of the reference
        ref_y = self.scans[REF]['Ys'][:, 0, 0]
        np.testing.assert_allclose(out['Ys'][120:200, 0, 0], ref_y[20:100], atol=1e-9)

    def test_an_offset_that_is_not_a_whole_number_of_layers(self):
        offsets = offsets_for((0.0, 0.0, 0.004))  # 0.4 of a layer
        out = tomo_merge.merge_masks(self.scans, offsets, reference=REF)
        np.testing.assert_allclose(np.diff(out['Ys'][:, 0, 0]), VS, atol=1e-9)
        self.assertGreater(iou(out['mask'][:270], truth_slices(out['mask'], self.truth)[:270]), 0.93)

    def test_shifting_by_a_fraction_of_a_voxel_does_not_grow_the_mask(self):
        square = np.zeros([1, N, N], bool)
        square[0, 10:20, 10:20] = True
        for shift in ((0.4, 0.4), (-0.3, 0.45), (0.0, 1.0), (2.0, -1.0)):
            moved = tomo_merge.shift_mask(square, np.array(shift))
            self.assertLessEqual(moved.sum(), square.sum() * 1.05, shift)  # not by an edge of partial voxels
            self.assertGreaterEqual(moved.sum(), square.sum() * 0.8, shift)
        np.testing.assert_array_equal(tomo_merge.shift_mask(square, np.zeros(2)), square)

    def test_overlapping_masks_are_blended_toward_the_one_whose_middle_it_is(self):
        # scan 0 has something only in its last layers that scan 1, which has the same height there,
        # does not see: near the end of scan 0 the merged mask follows scan 1
        blob = np.zeros([N, N], bool)
        blob[2:6, 2:6] = True  # outside the sample
        scans = [dict(s) for s in self.scans]
        mask0 = scans[0]['mask'].copy()
        mask0[-5:] |= blob
        scans[0]['mask'] = mask0
        out = tomo_merge.merge_masks(scans, self.offsets, reference=REF, register=False)
        self.assertTrue(mask0[-5:, 2:6, 2:6].all())  # it is in scan 0
        self.assertEqual(out['mask'][115:120, 2:6, 2:6].sum(), 0)  # and not in the merge, as scan 1 has none

    def test_two_masks(self):
        out = tomo_merge.merge_masks(self.scans[:2], self.offsets[:2], reference=1)
        self.assertEqual(out['mask'].shape[0], 220)

    def test_a_mask_that_does_not_overlap_cannot_be_registered(self):
        far = [dict(s) for s in self.scans[:2]]
        with self.assertRaisesRegex(ValueError, 'no usable slices in common'):
            tomo_merge.merge_masks(far, np.array([0.0, 5.0]), reference=0)
        # the same with the registration off: masks that do not reach each other are not merged
        with self.assertRaisesRegex(ValueError, 'gap of .* mm between'):
            tomo_merge.merge_masks(far, np.array([0.0, 5.0]), reference=0, register=False)

    def test_a_shift_that_is_too_big_stops_it(self):
        scans = [self.scans[0], make_scan(self.truth, 1, (9, 9))[0]]
        with self.assertRaisesRegex(ValueError, 'more than'):
            tomo_merge.merge_masks(scans, self.offsets[:2], reference=0, max_shift=5.0)

    def test_masks_that_do_not_fit_together(self):
        other = dict(self.scans[1], voxel_spacing=np.array(0.02))
        with self.assertRaisesRegex(ValueError, 'voxel spacing'):
            tomo_merge.check_masks([self.scans[0], other])
        small = dict(self.scans[1], mask=self.scans[1]['mask'][:, :-1])
        with self.assertRaisesRegex(ValueError, 'plane of the layers'):
            tomo_merge.check_masks([self.scans[0], small])
        moved = dict(self.scans[1], Xs=self.scans[1]['Xs'] + 0.5)
        with self.assertRaisesRegex(ValueError, 'different grid'):
            tomo_merge.check_masks([self.scans[0], moved])
        with self.assertRaisesRegex(ValueError, 'at least two'):
            tomo_merge.check_masks([self.scans[0]])


class TestOffsets(unittest.TestCase):

    def setUp(self):
        truth = make_sample()
        self.scans = [make_scan(truth, i)[0] for i in range(3)]

    def test_offsets_given_are_used_as_they_are(self):
        offs, ref, ramsz = tomo_merge.resolve_offsets(self.scans, offsets=[-0.9, 0.0, 0.9])
        np.testing.assert_allclose(offs, [-0.9, 0, 0.9])
        self.assertEqual(ref, 1)
        self.assertIsNone(ramsz)  # nothing says what the ramsz of the reference is

    def test_offsets_from_ramsz(self):
        # a scan at a lower ramsz has its sample lower, so its heights are shifted up to the reference
        offs, ref, ramsz = tomo_merge.resolve_offsets(self.scans, ramsz=[1.0, 0.1, -0.8])
        np.testing.assert_allclose(offs, [-0.9, 0.0, 0.9])
        self.assertEqual((ref, ramsz), (1, 0.1))

    def test_a_chosen_reference(self):
        offs, ref, ramsz = tomo_merge.resolve_offsets(self.scans, ramsz=[1.0, 0.1, -0.8], reference=0)
        np.testing.assert_allclose(offs, [0.0, 0.9, 1.8])
        self.assertEqual((ref, ramsz), (0, 1.0))

    def test_ramsz_recorded_in_the_masks(self):
        scans = [dict(s, ramsz=np.array(r)) for s, r in zip(self.scans, [1.0, 0.1, -0.8])]
        offs, ref, ramsz = tomo_merge.resolve_offsets(scans)
        np.testing.assert_allclose(offs, [-0.9, 0.0, 0.9])
        self.assertEqual(ramsz, 0.1)

    def test_ramsz_from_the_spec_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, 'spec.log')
            with open(log, 'w') as f:
                f.write('#F spec.log\n')
                for scan, ramsz in ((2, 1.0), (4, 0.1), (6, -0.8)):
                    f.write(f'#C Wed Sep 30 21:31:02 2026.  line 1 ramxx 6.10352e-06 ramsz {ramsz}.\n'
                            f'#S {scan}  flyscan ome 0 180 720 0.2\n#L ome  det_exp_out\n0.1  0.2\n')
            offs, ref, ramsz = tomo_merge.resolve_offsets(self.scans, spec_log=log, tomo_scans=[2, 4, 6])
            np.testing.assert_allclose(offs, [-0.9, 0.0, 0.9])
            with self.assertRaisesRegex(ValueError, 'scan 9'):
                tomo_merge.resolve_offsets(self.scans, spec_log=log, tomo_scans=[2, 4, 9])

    def test_what_is_given_wins_over_what_the_masks_record(self):
        scans = [dict(s, ramsz=np.array(r)) for s, r in zip(self.scans, [5.0, 5.0, 5.0])]  # wrong
        offs, ref, ramsz = tomo_merge.resolve_offsets(scans, ramsz=[1.0, 0.1, -0.8])
        np.testing.assert_allclose(offs, [-0.9, 0.0, 0.9])

    def test_the_ramsz_of_the_masks_is_only_passed_on_when_it_agrees_with_the_offsets(self):
        scans = [dict(s, ramsz=np.array(r)) for s, r in zip(self.scans, [1.0, 0.1, -0.8])]
        offs, ref, ramsz = tomo_merge.resolve_offsets(scans, offsets=[-0.9, 0.0, 0.9])
        self.assertEqual(ramsz, 0.1)
        with self.assertLogs(tomo_merge.logger, 'WARNING'):
            offs, ref, ramsz = tomo_merge.resolve_offsets(scans, offsets=[-0.5, 0.0, 0.5])
        self.assertIsNone(ramsz)

    def test_nothing_known_says_what_to_give(self):
        with self.assertRaisesRegex(ValueError, 'give offsets, ramsz'):
            tomo_merge.resolve_offsets(self.scans)

    def test_wrong_number_of_values(self):
        with self.assertRaisesRegex(ValueError, '2 offsets'):
            tomo_merge.resolve_offsets(self.scans, offsets=[0.0, 1.0])
        with self.assertRaisesRegex(ValueError, '2 ramsz'):
            tomo_merge.resolve_offsets(self.scans, ramsz=[0.0, 1.0])


class TestFilesAndCommand(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        truth = make_sample()
        self.files = []
        for i in range(3):
            scan = make_scan(truth, i, [(-2, 3), (0, 0), (2, -1)][i])[0]
            path = os.path.join(self.tmp.name, f'scan{i}_tomo_mask.npz')
            np.savez_compressed(path, **scan)
            self.files.append(path)
        self.out = os.path.join(self.tmp.name, 'all_tomo_mask.npz')
        self.offsets = offsets_for().tolist()

    def test_merged_file_has_the_keys_of_a_mask_and_the_record(self):
        tomo_merge.merge_mask_files(self.files, self.out, offsets=self.offsets, ramsz=[1.0, 0.1, -0.8])
        # the ramsz is only the "given" one, the offsets are the ones that put the scans in place
        with np.load(self.out) as f:
            self.assertEqual(sorted(f.files), sorted(['mask', 'Xs', 'Ys', 'Zs', 'voxel_spacing', 'ramsz',
                                                      'merge_offsets', 'merge_shifts_voxels', 'merge_spans_mm',
                                                      'merged_from']))
            self.assertEqual(f['mask'].dtype, bool)
            self.assertEqual(float(f['ramsz']), 0.1)
            self.assertEqual(float(f['voxel_spacing']), VS)
            self.assertEqual(f['mask'].shape, f['Ys'].shape)
            self.assertIn('scan1_tomo_mask.npz', str(f['merged_from']))

    def test_no_ramsz_is_recorded_when_it_is_not_known(self):
        with self.assertLogs(tomo_merge.logger, 'WARNING'):
            tomo_merge.merge_mask_files(self.files, self.out, offsets=self.offsets)
        with np.load(self.out) as f:
            self.assertNotIn('ramsz', f.files)

    def test_the_merged_ramsz_lets_the_nf_side_find_the_vertical_offset(self):
        from hexrd.hedm.grainmap.nfutil.utilities import resolve_vertical_motor_position
        tomo_merge.merge_mask_files(self.files, self.out, ramsz=[1.0, 0.1, -0.8])
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, 'spec.log')
            with open(log, 'w') as f:
                f.write('#F spec.log\n#C Wed Sep 30 21:31:02 2026.  line 1 ramxx 0 ramsz 0.6.\n'
                        '#S 8  flyscan ome 0 180 720 0.2\n#L ome  det_exp_out\n0.1  0.2\n')
            offset, source = resolve_vertical_motor_position(self.out, spec_log=log, nf_scan=8)
        self.assertEqual(source, 'spec log')
        self.assertAlmostEqual(offset, 0.6 - 0.1)  # the NF scan's ramsz minus that of the reference scan

    def test_figure_of_the_cross_sections_before_and_after(self):
        figure = os.path.join(self.tmp.name, 'sections.png')
        tomo_merge.merge_mask_files(self.files, self.out, offsets=self.offsets, figure=figure)
        with open(figure, 'rb') as f:
            self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')
        self.assertGreater(os.path.getsize(figure), 10000)
        with np.load(self.out) as f:
            np.testing.assert_allclose(f['merge_spans_mm'][:, 1] - f['merge_spans_mm'][:, 0],
                                       (np.array(LENGTHS) - 1) * VS)

    def test_no_figure_unless_asked_or_when_there_is_no_registration(self):
        figure = os.path.join(self.tmp.name, 'sections.png')
        tomo_merge.merge_mask_files(self.files, self.out, offsets=self.offsets)
        self.assertFalse(os.path.exists(figure))
        tomo_merge.merge_mask_files(self.files, self.out, offsets=self.offsets, register=False, figure=figure)
        self.assertFalse(os.path.exists(figure))  # it shows what the registration changes

    def test_the_command(self):
        parser = argparse.ArgumentParser()
        tomo_cli.configure_parser(parser.add_subparsers())
        args = parser.parse_args(['tomo', 'merge', *self.files, '-o', self.out,
                                  '--offsets', *[str(o) for o in self.offsets], '--register-slices', '3', '-q'])
        args.debug = False  # which hexrd's own parser sets
        args.func(args, parser)
        with np.load(self.out) as f:
            self.assertEqual(f['mask'].shape, (280, N, N))
        # the figure is next to the output unless it is turned off
        self.assertTrue(os.path.exists(os.path.splitext(self.out)[0] + '_merge.png'))
        logging.getLogger('hexrd').handlers.clear()

    def test_the_command_can_leave_out_the_figure_or_put_it_elsewhere(self):
        parser = argparse.ArgumentParser()
        tomo_cli.configure_parser(parser.add_subparsers())
        base = ['tomo', 'merge', *self.files, '-o', self.out, '--offsets', *[str(o) for o in self.offsets], '-q']
        args = parser.parse_args(base + ['--no-figure'])
        args.debug = False
        args.func(args, parser)
        self.assertFalse(os.path.exists(os.path.splitext(self.out)[0] + '_merge.png'))
        elsewhere = os.path.join(self.tmp.name, 'check.png')
        args = parser.parse_args(base + ['--figure', elsewhere])
        args.debug = False
        args.func(args, parser)
        self.assertTrue(os.path.exists(elsewhere))
        logging.getLogger('hexrd').handlers.clear()

    def test_the_command_reports_a_problem_and_exits(self):
        parser = argparse.ArgumentParser()
        tomo_cli.configure_parser(parser.add_subparsers())
        args = parser.parse_args(['tomo', 'merge', *self.files, '-o', self.out, '-q'])  # nothing says the offsets
        args.debug = False  # which hexrd's own parser sets
        with self.assertRaises(SystemExit):
            args.func(args, parser)
        self.assertFalse(os.path.exists(self.out))
        logging.getLogger('hexrd').handlers.clear()


if __name__ == '__main__':
    unittest.main()
