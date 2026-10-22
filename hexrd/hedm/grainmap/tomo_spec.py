"""Parse a SPEC log file into a list of scans.

For each scan, the following are extracted:
  #S <num> <command>   scan number and the spec command
  #L <names...>        column names (separated by two or more spaces)
  <no leading #>       numerical data rows

From the command line (see ``hexrd tomo spec -h``):
    hexrd tomo spec spec.log            # print a summary table
    hexrd tomo spec spec.log 25         # print details of scan 25
    hexrd tomo spec spec.log --tomo     # tomography scans + det_exp_out stats

As a module:
    from hexrd.hedm.grainmap.tomo_spec import parse_spec
    scans = parse_spec("spec.log")           # dict: scan number -> Scan
    s = scans[25]
    s.command, s.columns, s.data["ome"]
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np

_S_RE = re.compile(r"^#S\s+(\d+)\s*(.*)$")
_RETIGA_RE = re.compile(r"^Retiga #\d+ saved last image to (\S+?)\.?$")
# spec writes this #C comment just before the #S line of each tomography scan
_TOMO_MARK = "doing the tomography scan"
# The sample's vertical stage position (mm) is not in the scan header or columns. A macro logs it in a #C comment
# just before the scans taken at that position ("line 2 ramxx 6.1e-06 ramsz 0.35."), and "ramsz reset from -0.05 to 0."
# when the stage is redefined. It holds for the scans that follow, until the next such comment.
_NUM = r"-?[0-9.]+(?:[eE][-+]?\d+)?"
_RAMSZ_RE = re.compile(rf"\bramsz\s+({_NUM})\.?\s*$")
_RAMSZ_RESET_RE = re.compile(rf"\bramsz reset from {_NUM} to ({_NUM})\.?\s*$")


@dataclass
class Scan:
    number: int
    command: str
    date: str = ""
    columns: list = field(default_factory=list)
    comments: list = field(default_factory=list)  # #C lines within the scan
    is_tomo: bool = False       # preceded by the "doing the tomography scan" note
    retiga_file: str = ""       # path of the retiga .h5 saved during the scan
    ramsz: object = None        # vertical sample stage position (mm) at the scan, from the #C comments, or None
    data: dict = field(default_factory=dict)       # column name -> 1D array

    @property
    def npoints(self):
        return len(next(iter(self.data.values()))) if self.data else 0

    def as_array(self):
        """Data as an (npoints, ncolumns) array in column order."""
        return np.column_stack([self.data[c] for c in self.columns])


def _ramsz_in(line):
    """The ramsz value a #C comment sets, or None."""
    for regex in (_RAMSZ_RESET_RE, _RAMSZ_RE):
        m = regex.search(line)
        if m:
            return float(m.group(1).rstrip("."))
    return None


def ramsz_of_scan(path, scan):
    """Vertical sample stage position (mm) in effect at scan number ``scan``, or None if the log does not say.

    Reads the log only as far as the scan. See the note above ``_RAMSZ_RE``.
    """
    current = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("#C"):
                value = _ramsz_in(line)
                if value is not None:
                    current = value
            else:
                m = _S_RE.match(line)
                if m and int(m.group(1)) == scan:
                    return current
    raise ValueError(f"scan {scan} not found in {path}")


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
    ramsz = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if line.startswith("#C"):
                    value = _ramsz_in(line)
                    if value is not None:
                        ramsz = value
                m = _S_RE.match(line)
                if m:
                    _finalize(scan, rows)
                    scan = Scan(int(m.group(1)), m.group(2).strip(), is_tomo=tomo_pending, ramsz=ramsz)
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


def write_missing_slots(scans, out_dir):
    """Write <retiga stem>_missing_slots.txt (one 0-based slot per line) for every tomo scan.

    Slots come from det_exp_out. Returns [(scan number, number dropped, path)].
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for s in (s for s in scans.values() if s.is_tomo):
        stem = os.path.splitext(os.path.basename(s.retiga_file))[0] or f"scan_{s.number}"
        slots = dropped_slots(s)
        path = os.path.join(out_dir, f"{stem}_missing_slots.txt")
        with open(path, "w") as f:
            f.write("".join(f"{i}\n" for i in slots))
        written.append((s.number, len(slots), path))
    return written


def tomo_summary(scans):
    """Lines of text listing the tomography scans with det_exp_out statistics."""
    tomo = [s for s in scans.values() if s.is_tomo]
    lines = [f"{len(tomo)} tomography scans",
             f"{'scan':>5} {'npts':>5} {'min':>8} {'mean':>8} {'max':>8}  {'retiga file':<16} command"]
    for s in tomo:
        d = s.data["det_exp_out"]
        lines.append(f"{s.number:>5} {s.npoints:>5} {d.min():>8.4f} {d.mean():>8.4f} "
                     f"{d.max():>8.4f}  {s.retiga_file.split('/')[-1]:<16} {s.command}")
    return lines


def write_det_exp_out_csv(scans, path):
    """Write per-point det_exp_out for every tomo scan."""
    with open(path, "w") as f:
        f.write("scan,point,ome,det_exp_out\n")
        for s in (s for s in scans.values() if s.is_tomo):
            for i, (o, d) in enumerate(zip(s.data["ome"], s.data["det_exp_out"])):
                f.write(f"{s.number},{i},{o:.6f},{d:.9g}\n")


def scan_table(scans):
    """Lines of text listing every scan."""
    lines = [f"{'scan':>5} {'npts':>6}  command"]
    lines += [f"{n:>5} {s.npoints:>6}  {s.command}" for n, s in scans.items()]
    return lines


def scan_details(scan):
    """Lines of text describing one scan."""
    lines = [f"Scan {scan.number}: {scan.command}",
             f"Date: {scan.date}",
             f"Tomo scan: {scan.is_tomo}   Retiga file: {scan.retiga_file or '-'}",
             f"{scan.npoints} points, {len(scan.columns)} columns:",
             "  " + ", ".join(scan.columns)]
    lines += ["  #C " + c for c in scan.comments]
    return lines


def vertical_offset(spec_log, tomo_scan, nf_scan):
    """(ramsz of the tomography scan, ramsz of the NF scan, offset) in mm, from the spec log.

    The offset is ``ramsz_nf - ramsz_tomo``: add it to the heights of the tomography mask to get heights in the
    NF layer (the NF config ``vertical_motor_position``). Positive ramsz moves the sample up, so a sample that is
    higher by ``z`` has its features ``z`` higher in the lab. Raises ValueError if the log has no ramsz for a scan.
    """
    z_tomo = ramsz_of_scan(spec_log, tomo_scan)
    z_nf = ramsz_of_scan(spec_log, nf_scan)
    missing = sorted({n for n, z in ((tomo_scan, z_tomo), (nf_scan, z_nf)) if z is None})
    if missing:
        raise ValueError(f"{spec_log} has no vertical position (ramsz) for scan(s) {missing}")
    return z_tomo, z_nf, z_nf - z_tomo


def vertical_offset_report(spec_log, tomo_scan, nf_scan, mask=None, span=None):
    """Lines of text giving the vertical offset, and with ``mask`` and ``span`` the mask layers the NF run would use.

    ``span`` is the NF vertical span (``desired_vertical_span``, mm, measured from the NF beam center). A mask
    layer is used if its height plus the offset is at least ``span[0]`` and below ``span[1]``, as in
    ``nfutil.generate_test_coordinates``.
    """
    z_tomo, z_nf, offset = vertical_offset(spec_log, tomo_scan, nf_scan)
    lines = [f"tomography scan {tomo_scan}: ramsz {z_tomo:g} mm",
             f"NF scan {nf_scan}: ramsz {z_nf:g} mm",
             f"vertical_motor_position = ramsz_nf - ramsz_tomo = {offset:+g} mm",
             "  (added to the mask heights Ys to get NF heights; positive ramsz moves the sample up)"]
    if mask is not None and span is not None:
        d = np.load(mask)
        ys = np.asarray(d["Ys"])[:, 0, 0] if np.asarray(d["Ys"]).ndim == 3 else np.asarray(d["Ys"])
        nf_heights = ys + offset
        used = np.flatnonzero((nf_heights >= span[0]) & (nf_heights < span[1]))
        lines.append(f"mask layers: {len(ys)}, heights {ys.min():+.4f} to {ys.max():+.4f} mm in the mask, "
                     f"{nf_heights.min():+.4f} to {nf_heights.max():+.4f} mm for this NF scan")
        if len(used):
            ids = np.asarray(d["layer_indices"])[used] if "layer_indices" in d.files else used
            lines.append(f"NF vertical span [{span[0]:g}, {span[1]:g}) uses {len(used)} mask layer(s): "
                         f"positions {used.tolist()}, layer ids {np.asarray(ids).tolist()}, "
                         f"NF heights {np.round(nf_heights[used], 4).tolist()}")
        else:
            lines.append(f"NF vertical span [{span[0]:g}, {span[1]:g}) matches NO mask layer: this mask does not "
                         "cover the NF layer. Use the mask of a tomography scan taken closer to it.")
        if "ramsz" in d.files and np.isfinite(float(d["ramsz"])) and abs(float(d["ramsz"]) - z_tomo) > 1e-6:
            lines.append(f"WARNING: the mask records ramsz {float(d['ramsz']):g} but scan {tomo_scan} is at {z_tomo:g}")
    return lines
