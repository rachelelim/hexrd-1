#!/usr/bin/env python3
"""Parse a SPEC log file into a list of scans.

For each scan, the following are extracted:
  #S <num> <command>   scan number and the spec command
  #L <names...>        column names (separated by two or more spaces)
  <no leading #>       numerical data rows

Usage:
    python parse_spec.py spec.log            # print a summary table
    python parse_spec.py spec.log 25         # print details of scan 25
    python parse_spec.py spec.log --tomo     # tomography scans + det_exp_out stats
    python parse_spec.py spec.log --tomo --csv det_exp_out.csv
    python parse_spec.py spec.log --missing-slots-dir missing_slots

As a module:
    from parse_spec import parse_spec
    scans = parse_spec("spec.log")           # dict: scan number -> Scan
    s = scans[25]
    s.command, s.columns, s.data["ome"]
"""
import argparse
import os
import re
from dataclasses import dataclass, field

import numpy as np

_S_RE = re.compile(r"^#S\s+(\d+)\s*(.*)$")
_RETIGA_RE = re.compile(r"^Retiga #\d+ saved last image to (\S+?)\.?$")
# spec writes this #C comment just before the #S line of each tomography scan
_TOMO_MARK = "doing the tomography scan"


@dataclass
class Scan:
    number: int
    command: str
    date: str = ""
    columns: list = field(default_factory=list)
    comments: list = field(default_factory=list)  # #C lines within the scan
    is_tomo: bool = False       # preceded by the "doing the tomography scan" note
    retiga_file: str = ""       # path of the retiga .h5 saved during the scan
    data: dict = field(default_factory=dict)       # column name -> 1D array

    @property
    def npoints(self):
        return len(next(iter(self.data.values()))) if self.data else 0

    def as_array(self):
        """Data as an (npoints, ncolumns) array in column order."""
        return np.column_stack([self.data[c] for c in self.columns])


def dropped_slots(scan, frac=0.5):
    """Trigger slots (0-based) whose det_exp_out fell below frac * median.

    A dropped frame leaves det_exp_out near zero (~0.007 vs ~0.17-0.21 for
    a normal one). Only triggers inside the scan's recorded points can be
    seen, so frames dropped after the last point are not detected.
    """
    d = scan.data["det_exp_out"]
    return np.flatnonzero(d < frac * np.median(d))


def _finalize(scan, rows):
    """Attach the collected data rows to a scan as named columns."""
    if scan is None:
        return
    if rows:
        ncol = len(scan.columns)
        # Drop ragged rows (e.g. a partially written last line of an
        # aborted scan) rather than failing on the whole file.
        rows = [r for r in rows if len(r) == ncol]
    if rows:
        arr = np.array(rows, dtype=float)
        scan.data = {name: arr[:, i] for i, name in enumerate(scan.columns)}
    else:
        scan.data = {name: np.array([]) for name in scan.columns}


def parse_spec(path):
    """Return {scan number: Scan} for every #S block in the file."""
    scans = {}
    scan = None
    rows = []
    tomo_pending = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                m = _S_RE.match(line)
                if m:
                    _finalize(scan, rows)
                    scan = Scan(int(m.group(1)), m.group(2).strip(), is_tomo=tomo_pending)
                    tomo_pending = False
                    scans[scan.number] = scan
                    rows = []
                elif _TOMO_MARK in line:
                    tomo_pending = True
                elif scan is None:
                    continue  # file header before the first scan
                elif line.startswith("#L"):
                    # Names are separated by 2+ spaces (single spaces
                    # can't occur inside names, but this is spec's format).
                    scan.columns = re.split(r"\s{2,}|\s+", line[2:].strip())
                elif line.startswith("#D"):
                    scan.date = line[2:].strip()
                elif line.startswith("#C"):
                    scan.comments.append(line[2:].strip())
            elif scan is not None:
                try:
                    rows.append([float(x) for x in line.split()])
                except ValueError:
                    m = _RETIGA_RE.match(line.strip())
                    if m:
                        scan.retiga_file = m.group(1)
    _finalize(scan, rows)
    return scans


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logfile")
    ap.add_argument("scan", nargs="?", type=int, help="show details of this scan")
    ap.add_argument("--tomo", action="store_true",
                    help="list tomography scans with det_exp_out statistics")
    ap.add_argument("--missing-slots-dir", metavar="DIR",
                    help="write a <retiga stem>_missing_slots.txt (one 0-based "
                         "slot per line) for every tomo scan, from det_exp_out")
    ap.add_argument("--csv", metavar="OUT.csv",
                    help="with --tomo: write per-point det_exp_out for every tomo scan")
    args = ap.parse_args()

    scans = parse_spec(args.logfile)
    if args.missing_slots_dir:
        os.makedirs(args.missing_slots_dir, exist_ok=True)
        for s in (s for s in scans.values() if s.is_tomo):
            stem = os.path.splitext(os.path.basename(s.retiga_file))[0] \
                or f"scan_{s.number}"
            slots = dropped_slots(s)
            path = os.path.join(args.missing_slots_dir, f"{stem}_missing_slots.txt")
            with open(path, "w") as f:
                f.write("".join(f"{i}\n" for i in slots))
            print(f"scan {s.number:>3}  {len(slots):>3} dropped  -> {path}")
    elif args.tomo:
        tomo = [s for s in scans.values() if s.is_tomo]
        print(f"{len(tomo)} tomography scans")
        print(f"{'scan':>5} {'npts':>5} {'min':>8} {'mean':>8} {'max':>8}  {'retiga file':<16} command")
        for s in tomo:
            d = s.data["det_exp_out"]
            print(f"{s.number:>5} {s.npoints:>5} {d.min():>8.4f} {d.mean():>8.4f} "
                  f"{d.max():>8.4f}  {s.retiga_file.split('/')[-1]:<16} {s.command}")
        if args.csv:
            with open(args.csv, "w") as f:
                f.write("scan,point,ome,det_exp_out\n")
                for s in tomo:
                    for i, (o, d) in enumerate(zip(s.data["ome"], s.data["det_exp_out"])):
                        f.write(f"{s.number},{i},{o:.6f},{d:.9g}\n")
            print(f"wrote {args.csv}")
    elif args.scan is None:
        print(f"{len(scans)} scans in {args.logfile}")
        print(f"{'scan':>5} {'npts':>6}  command")
        for n, s in scans.items():
            print(f"{n:>5} {s.npoints:>6}  {s.command}")
    else:
        s = scans[args.scan]
        print(f"Scan {s.number}: {s.command}")
        print(f"Date: {s.date}")
        print(f"Tomo scan: {s.is_tomo}   Retiga file: {s.retiga_file or '-'}")
        print(f"{s.npoints} points, {len(s.columns)} columns:")
        print("  " + ", ".join(s.columns))
        for c in s.comments:
            print("  #C", c)


if __name__ == "__main__":
    main()
