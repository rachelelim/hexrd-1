"""run_tomo_pipeline.locate: finding the dark, bright and tomo h5 files (no data needed; the files are empty)."""
import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

TOMO = Path(__file__).resolve().parent.parent / 'tomo'
sys.path.insert(0, str(TOMO))
import run_tomo_pipeline as rtp  # noqa: E402
from parse_spec import parse_spec  # noqa: E402

SPEC = """#F spec.log

#S 21  flyscan ome 0 10 20 0.2
#L ome  det_exp_out
0.1  0.2
Retiga #0 saved last image to /not/here/retiga_000021.h5.

#S 22  flyscan ome 0 10 20 0.2
#L ome  det_exp_out
0.1  0.2
Retiga #0 saved last image to /not/here/retiga_000022.h5.

#C doing the tomography scan
#S 23  flyscan ome 0 180 720 0.2
#L ome  det_exp_out
0.1  0.2
Retiga #0 saved last image to /not/here/retiga_000023.h5.
"""


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return path


class TestLocate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        log = self.dir / 'spec.log'
        log.write_text(SPEC)
        self.scans = parse_spec(log)
        self.sample = self.dir / 'sample'
        self.dark = touch(self.sample / '21' / 'nf' / 'retiga_900021.h5')
        self.bright = touch(self.sample / '22' / 'nf' / 'retiga_900022.h5')
        self.tomo = touch(self.sample / '23' / 'nf' / 'retiga_900023.h5')

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **kw):
        base = dict(scan=23, tomo_dir=str(self.dir / 'flat'), dark=None, bright=None, tomo=None, sample_dir=None, dry_run=False)
        return argparse.Namespace(**{**base, **kw})

    def test_sample_dir_finds_the_files_by_scan_number(self):
        names = rtp.locate(self.args(sample_dir=str(self.sample)), self.scans)
        self.assertEqual((names['dark'], names['bright'], names['tomo']), (self.dark, self.bright, self.tomo))

    def test_names_in_the_spec_log_are_not_needed(self):
        for n in (21, 22):
            self.scans[n].retiga_file = ''
        names = rtp.locate(self.args(sample_dir=str(self.sample)), self.scans)
        self.assertEqual(names['dark'], self.dark)

    def test_spec_name_picks_one_of_several(self):
        named = touch(self.tomo.parent / 'retiga_000023.h5')
        touch(self.tomo.parent / 'retiga_900999.h5')
        self.assertEqual(rtp.locate(self.args(sample_dir=str(self.sample)), self.scans)['tomo'], named)

    def test_several_files_and_no_name_to_choose_by_stops(self):
        touch(self.bright.parent / 'retiga_900999.h5')
        self.scans[22].retiga_file = '/x/retiga_zzz.h5'
        with self.assertRaises(SystemExit) as ctx:
            rtp.locate(self.args(sample_dir=str(self.sample)), self.scans)
        self.assertIn('bright: 2 retiga_*.h5 files', str(ctx.exception))

    def test_missing_folder_and_empty_folder_stop(self):
        self.tomo.unlink()
        with self.assertRaisesRegex(SystemExit, 'tomo: no retiga_'):
            rtp.locate(self.args(sample_dir=str(self.sample)), self.scans)
        self.dark.unlink()
        self.dark.parent.rmdir()
        self.dark.parent.parent.rmdir()
        with self.assertRaisesRegex(SystemExit, 'dark: .*does not exist'):
            rtp.locate(self.args(sample_dir=str(self.sample)), self.scans)

    def test_explicit_files_win(self):
        other = touch(self.dir / 'other.h5')
        names = rtp.locate(self.args(sample_dir=str(self.sample), dark=str(other), tomo=str(other)), self.scans)
        self.assertEqual((names['dark'], names['tomo'], names['bright']), (other, other, self.bright))

    def test_sample_dir_wins_over_tomo_dir(self):
        flat = self.dir / 'flat'
        for n in (21, 22, 23):
            touch(flat / f'retiga_0000{n}.h5')
        names = rtp.locate(self.args(sample_dir=str(self.sample), tomo_dir=str(flat)), self.scans)
        self.assertEqual(names['tomo'], self.tomo)

    def test_without_sample_dir_the_spec_names_are_looked_for_in_tomo_dir(self):
        flat = self.dir / 'flat'
        for n in (21, 22, 23):
            touch(flat / f'retiga_0000{n}.h5')
        names = rtp.locate(self.args(tomo_dir=str(flat)), self.scans)
        self.assertEqual({k: v.name for k, v in names.items()},
                         {'dark': 'retiga_000021.h5', 'bright': 'retiga_000022.h5', 'tomo': 'retiga_000023.h5'})

    def test_files_that_do_not_exist_stop_unless_dry_run(self):
        with self.assertRaisesRegex(SystemExit, 'not found'):
            rtp.locate(self.args(), self.scans)
        self.assertEqual(len(rtp.locate(self.args(dry_run=True), self.scans)), 3)


if __name__ == '__main__':
    unittest.main()
