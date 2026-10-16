#!/usr/bin/env python
"""
find_missing_frames.py - locate dropped frames in an areaDetector HDF5 stack
and build the projection-angle list for tomographic reconstruction.

How it works
------------
areaDetector writes per-frame EPICS timestamps and a driver frame counter to
  /entry/instrument/NDAttributes/{NDArrayEpicsTSSec, NDArrayEpicsTSnSec, NDArrayUniqueId}
For a scan triggered at a fixed period, the interval between consecutive
retained frames is k * period, where k - 1 is the number of triggers missed
between them.  Counting the gaps assigns every retained frame its original
trigger index ("slot").  Gaps in NDArrayUniqueId additionally show frames the
camera produced but that were lost before reaching the file.

Angles are then  theta = start_deg + deg_per_slot * slot  (or, with
--span-deg, the first-to-last span is forced to that many degrees).

Limitations: triggers missed before the first or after the last retained
frame are invisible to this method (compare the slot count with --expected
and with a complete reference scan if that matters).

Examples
--------
  python find_missing_frames.py retiga_012547.h5
  python find_missing_frames.py retiga_012547.h5 --expected 721 --deg-per-slot 0.25
  python find_missing_frames.py scan.h5 --span-deg 180 --out-prefix results/scan23
  python find_missing_frames.py scan.h5 --dataset /entry/data/data --attr-group /entry/instrument/NDAttributes

Outputs (prefix defaults to the h5 path without extension)
  <prefix>_frame_angles.csv    frame_index, unique_id, timestamp_s, slot, missing_before, theta_deg
  <prefix>_theta_deg.npy       angle per retained frame, in stack order (feed to the reconstruction)
  <prefix>_missing_slots.txt   original trigger indices with no frame
  <prefix>_gap_diagnostics.png unless --no-plot
Exit status is 0 when no frames are missing, 1 when some are, 2 on error.
"""
import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  registers external compression filters if present
except ImportError:
    pass


def read_metadata(h5path, dataset, attr_group):
    with h5py.File(h5path, "r") as f:
        if dataset not in f:
            raise KeyError(f"dataset {dataset} not found in {h5path}")
        nframes = f[dataset].shape[0]
        if attr_group not in f:
            raise KeyError(f"attribute group {attr_group} not found in {h5path}; "
                           "this file has no per-frame timestamps")
        g = f[attr_group]
        for key in ("NDArrayEpicsTSSec", "NDArrayEpicsTSnSec"):
            if key not in g:
                raise KeyError(f"{attr_group}/{key} missing")
        t = g["NDArrayEpicsTSSec"][()].astype(np.float64) \
            + g["NDArrayEpicsTSnSec"][()].astype(np.float64) * 1e-9
        uid = g["NDArrayUniqueId"][()] if "NDArrayUniqueId" in g else None
    if len(t) != nframes:
        raise RuntimeError(f"{len(t)} timestamps but {nframes} frames in {dataset}")
    return t, uid, nframes


def slots_from_timestamps(t, period=None, tol=0.35):
    """Original trigger index per frame, from the timestamp intervals."""
    dt = np.diff(t)
    if len(dt) == 0:
        return np.zeros(1, int), period or 0.0, dt
    if period is None:
        period = float(np.median(dt))
    ratio = dt / period
    k = np.round(ratio)
    off = np.abs(ratio - k)
    bad = np.flatnonzero((off > tol) | (k < 1))
    if len(bad):
        detail = ", ".join(f"frame {i}->{i+1}: {ratio[i]:.3f}" for i in bad[:10])
        raise RuntimeError(
            "Some intervals are not close to an integer number of trigger periods "
            f"(period {period:.4f} s, tolerance {tol}): {detail}"
            + (" ..." if len(bad) > 10 else "")
            + "\nIf the trigger period is known, pass it with --period; "
            "if the scan was not triggered at a fixed period this method does not apply.")
    slot = np.concatenate([[0], np.cumsum(k.astype(int))])
    return slot, period, dt


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5file")
    ap.add_argument("--dataset", default="/entry/data/data")
    ap.add_argument("--attr-group", default="/entry/instrument/NDAttributes")
    ap.add_argument("--expected", type=int, default=None,
                    help="number of frames the scan should have produced")
    ap.add_argument("--deg-per-slot", type=float, default=0.25,
                    help="angular step per trigger (default 0.25)")
    ap.add_argument("--span-deg", type=float, default=None,
                    help="force first->last retained frame to span this many degrees "
                         "(overrides --deg-per-slot)")
    ap.add_argument("--start-deg", type=float, default=0.0,
                    help="angle of the first retained frame (default 0)")
    ap.add_argument("--period", type=float, default=None,
                    help="trigger period in seconds (default: median frame interval)")
    ap.add_argument("--tol", type=float, default=0.35,
                    help="allowed deviation of interval/period from an integer (default 0.35)")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    h5path = Path(args.h5file)
    if not h5path.is_file():
        print(f"ERROR: {h5path} not found", file=sys.stderr)
        return 2
    prefix = Path(args.out_prefix) if args.out_prefix else h5path.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    try:
        t, uid, nframes = read_metadata(h5path, args.dataset, args.attr_group)
        slot, period, dt = slots_from_timestamps(t, args.period, args.tol)
    except (KeyError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    nslots = int(slot[-1]) + 1
    missing = np.setdiff1d(np.arange(nslots), slot)
    gap_sizes = np.diff(slot) - 1
    multi = np.flatnonzero(gap_sizes > 1)

    if args.span_deg is not None:
        deg_per_slot = args.span_deg / slot[-1] if slot[-1] else 0.0
    else:
        deg_per_slot = args.deg_per_slot
    theta = args.start_deg + deg_per_slot * slot

    print(f"file: {h5path}")
    print(f"frames in file: {nframes}")
    print(f"trigger slots spanned (first to last frame): {nslots}")
    print(f"missing frames inside that span: {len(missing)}"
          + (f" ({len(multi)} gaps of more than one frame)" if len(multi) else ""))
    if args.expected is not None:
        print(f"expected frames: {args.expected}; frames + detected gaps = {nframes + len(missing)}")
        extra = nslots - args.expected
        if extra > 0:
            print(f"  NOTE: {extra} more trigger slots than expected; the scan issued extra "
                  "triggers, or the expected count is off")
        elif extra < 0:
            print(f"  NOTE: {-extra} fewer slots than expected; triggers were probably missed "
                  "before the first or after the last retained frame (undetectable here)")
    if uid is not None:
        uid_gaps = np.flatnonzero(np.diff(uid) != 1)
        if len(uid_gaps):
            print(f"NDArrayUniqueId has {len(uid_gaps)} jumps: frames were lost AFTER the "
                  f"camera driver (after retained frames {uid_gaps.tolist()[:20]})")
        elif len(missing):
            print("NDArrayUniqueId contiguous: the camera never produced the missing frames "
                  "(missed triggers)")
        else:
            print("NDArrayUniqueId contiguous")
    print(f"trigger period: {period:.4f} s; first-to-last span: {t[-1]-t[0]:.3f} s")
    print(f"angle: {args.start_deg:g} + {deg_per_slot:.5f} deg * slot  ->  "
          f"{theta[0]:.3f} to {theta[-1]:.3f} deg")
    if len(missing):
        print("missing trigger slots (0-based):")
        print("  " + ", ".join(map(str, missing.tolist())))
        print("retained frame index immediately before each gap:")
        print("  " + ", ".join(str(int(np.flatnonzero(slot < m)[-1])) for m in missing))

    csv_path = prefix.parent / f"{prefix.name}_frame_angles.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_index", "unique_id", "timestamp_s", "slot",
                    "missing_before", "theta_deg"])
        for i in range(nframes):
            w.writerow([i, int(uid[i]) if uid is not None else "", f"{t[i]:.6f}",
                        int(slot[i]), int(slot[i] - i), f"{theta[i]:.5f}"])
    np.save(prefix.parent / f"{prefix.name}_theta_deg.npy", theta)
    np.savetxt(prefix.parent / f"{prefix.name}_missing_slots.txt", missing, fmt="%d")
    written = [csv_path.name, f"{prefix.name}_theta_deg.npy", f"{prefix.name}_missing_slots.txt"]

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(12, 7))
            ax[0].plot(np.arange(nframes - 1), dt / period, ".", ms=3)
            for lev in range(1, int(np.ceil((dt / period).max())) + 1):
                ax[0].axhline(lev, color="gray", lw=0.5)
            ax[0].set_ylabel("interval / trigger period")
            ax[0].set_xlabel("retained frame index")
            ax[0].set_title(f"{h5path.name}: {nframes} frames, {len(missing)} missing in {nslots} slots")
            ax[1].plot(np.arange(nframes), slot - np.arange(nframes), drawstyle="steps-post")
            ax[1].set_ylabel("missing frames before this frame")
            ax[1].set_xlabel("retained frame index")
            fig.tight_layout()
            fig.savefig(prefix.parent / f"{prefix.name}_gap_diagnostics.png", dpi=120)
            written.append(f"{prefix.name}_gap_diagnostics.png")
        except ImportError:
            print("matplotlib not available, skipping plot")
    print("wrote: " + ", ".join(written) + f"  (in {prefix.parent})")
    return 1 if len(missing) else 0


if __name__ == "__main__":
    sys.exit(main())
