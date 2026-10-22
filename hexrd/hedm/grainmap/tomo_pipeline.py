"""Tomography mask pipeline for one spec scan (``hexrd tomo pipeline``).

Stages (run in this order; a stage is skipped if its output exists, unless ``force``):
  setup   find the dark, bright and tomo files from the spec log and write <run_dir>/tomo_config.yml
  missing find dropped frames from the h5 timestamps  -> <run_dir>/<retiga>_missing_slots.txt
  recon   ``tomo_reconstruct.reconstruct`` (spec angles, missing slots, powder removal) -> <name>_<step>um_tomo_mask.npz
  qc      per-layer report: fill, layer-to-layer change, bright-field clipping; suggests bad layers
  interp  interpolate the layer ranges given in ``interp``           -> <name>_<step>um_tomo_mask_interp.npz
  smooth  smooth the (interpolated) mask                              -> ..._smooth.npz

The scan is described by the yml (nf_config format): ``tomography: spec: {log, scan, tomo_dir}``,
``input_files: detector_file``, ``analysis_name``, ``tomography: reconstruction:`` (``v_bnds``,
``voxel_spacing``, the grid) and ``tomography: processing:``. Anything the yml leaves out comes from
``DEFAULT_CONFIG``. The three h5 files are found, in order of preference: from ``tomography: dark / bright /
tomo_images: filename`` if set; else, if ``tomography: spec: sample_dir`` is set, as the retiga file in
<sample_dir>/<scan>/nf/ of the dark (scan - 2), bright (scan - 1) and tomo scan; else as the files named in the
spec log, looked for in ``tomography: spec: tomo_dir``. The run's complete config is written to
<run_dir>/tomo_config.yml, with the three h5 filenames and ``num_images`` filled in.
"""
import contextlib
import copy
import json
import logging
import re
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import yaml

from hexrd.hedm import nf_config
from hexrd.hedm.grainmap.tomo_frames import find_missing_frames
from hexrd.hedm.grainmap.tomo_options import STAGES, PipelineOptions, ReconstructOptions, mask_file_name
from hexrd.hedm.grainmap.tomo_masks import interpolate_mask_file, parse_ranges, smooth_mask_file
from hexrd.hedm.grainmap.tomo_reconstruct import load_instrument, reconstruct
from hexrd.hedm.grainmap.tomo_spec import dropped_slots, parse_spec

logger = logging.getLogger(__name__)

SAT = 4090                                                 # 12-bit detector ceiling is 4095

DEFAULT_CONFIG = """
analysis_name: tomo
output_directory: "./"
input_files:
  detector_file: null
output_plot_check: False
experiment:
  beam_energy: 61.332
  beam_vertical_span: [-0.35, 0.35]
tomography:
  img_stem: nf_
  filetype: h5
  num_digits: 6
  ome_start: 0.0
  ome_end: 180
  reconstruction:
    voxel_spacing: 0.01
    v_bnds: [-0.35, 0.35]
    x_center: 0
    z_center: 0
    x_length: 1.5
    z_length: 1.5
  processing:
    recon_thresh: 0.0002
    noise_obj_size: 300
    min_hole_size: 500
    erosion_iter: 2
    dilation_iter: 2
multiprocessing:
  num_cpus: half
  chunk_size: -1
"""


class PipelineError(RuntimeError):
    """A pipeline stage cannot continue; the message says why."""


def log(*a):
    logger.info(' '.join(str(x) for x in a))


@contextlib.contextmanager
def stage_log(path):
    """Also write the hexrd log to ``path`` while the block runs."""
    handler = logging.FileHandler(path, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    root = logging.getLogger('hexrd')
    old_level = root.level
    if root.getEffectiveLevel() > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()


def mask_name(analysis, layers, voxel_spacing):
    """The name of the mask file that the recon stage writes (see ``tomo_options.mask_file_name``)."""
    if layers is None:
        return mask_file_name(analysis, voxel_spacing)
    return mask_file_name(f'{analysis}_layer{layers[0]}' if len(layers) == 1 else f'{analysis}_{len(layers)}layers',
                          voxel_spacing)


def spec_step(scan):
    for c in scan.comments:
        m = re.search(r'rams pulse step:\s*([-0-9.eE+]+?)\.?\s*$', c)
        if m:
            return float(m.group(1))
    return None


def find_in_scan_dir(sample_dir, scan_number, role, spec_name=None):
    """The retiga h5 file of a scan in <sample_dir>/<scan>/nf/.

    If the folder holds several, the one named in the spec log is used; otherwise they are ambiguous.
    """
    folder = Path(sample_dir) / str(scan_number) / 'nf'
    if not folder.is_dir():
        raise PipelineError(f'{role}: {folder} does not exist (sample_dir {sample_dir}, scan {scan_number})')
    files = sorted(folder.glob('retiga_*.h5'))
    if spec_name and (folder / spec_name).exists():
        return folder / spec_name
    if len(files) == 1:
        return files[0]
    if not files:
        raise PipelineError(f'{role}: no retiga_*.h5 file in {folder}')
    raise PipelineError(f'{role}: {len(files)} retiga_*.h5 files in {folder} ({", ".join(f.name for f in files)}) and the '
                        'spec log does not say which; set the file in the yml (tomography: dark / bright / tomo_images: filename)')


def locate(scan_number, scans, tomo_dir, dark=None, bright=None, tomo=None, dry_run=False, sample_dir=None):
    """(dark, bright, tomo) retiga h5 paths for the tomo scan.

    The dark is scan - 2 and the bright is scan - 1. A file given in ``dark``, ``bright`` or ``tomo`` is used as is.
    Otherwise, with ``sample_dir``, each is the retiga file in <sample_dir>/<scan>/nf/; without it, each is the file
    named in the spec log, looked for in ``tomo_dir`` (and the numbers are checked against the log).
    """
    s = scans[scan_number]
    if not s.is_tomo:
        log(f'WARNING: scan {scan_number} was not preceded by the "doing the tomography scan" note')

    given = {'dark': dark, 'bright': bright, 'tomo': tomo}
    names = {}
    for role, n in (('dark', scan_number - 2), ('bright', scan_number - 1), ('tomo', scan_number)):
        if given[role]:
            names[role] = Path(given[role])
            continue
        sc = scans.get(n)
        spec_name = Path(sc.retiga_file).name if sc is not None and sc.retiga_file else None
        if sample_dir:
            names[role] = find_in_scan_dir(sample_dir, n, role, spec_name)
        elif spec_name:
            names[role] = Path(tomo_dir) / spec_name
        else:
            raise PipelineError(f'scan {n} ({role}) has no retiga file in spec.log')

    if not sample_dir and not any(given.values()):
        def num(sc):
            return int(re.search(r'retiga_(\d+)', sc.retiga_file).group(1))
        try:
            consecutive = (num(scans[scan_number - 2]) + 1 == num(scans[scan_number - 1])
                           and num(scans[scan_number - 1]) + 1 == num(s))
        except (KeyError, AttributeError):
            consecutive = False
        if not consecutive:
            log('WARNING: dark/bright/tomo retiga numbers are not consecutive; check the dark and bright filenames in the yml')
    missing = [str(p) for p in names.values() if not p.exists()]
    if missing and not dry_run:
        raise PipelineError('h5 file(s) not found: ' + ', '.join(missing))
    return names


def check_dark_bright(names):
    import h5py
    import hdf5plugin  # noqa: F401
    med = {}
    for k in ('dark', 'bright'):
        with h5py.File(names[k]) as f:
            med[k] = float(np.median(f['entry/data/data'][0]))
    log(f'frame-0 median counts: dark {med["dark"]:.0f}, bright {med["bright"]:.0f}')
    if not med['dark'] < med['bright']:
        raise PipelineError('dark and bright look swapped (dark median >= bright); set the dark and bright '
                            'filenames in the yml to override')


def run_config(cfg, names, scan, spec_log, tomo_dir, analysis, run_dir, sample_dir=None):
    """The config dict of this run: the defaults, then the user's yml, then what the spec log determines."""
    base = yaml.safe_load(DEFAULT_CONFIG)
    d = nf_config.utils.merge_dicts(base, copy.deepcopy(cfg._cfg))
    d['analysis_name'] = analysis
    d['output_directory'] = str(run_dir)
    det = d.get('input_files', {}).get('detector_file')
    if not det:
        raise PipelineError('input_files: detector_file must be set in the yml')
    d['input_files']['detector_file'] = str(Path(det).resolve())
    t = d['tomography']
    for key, role in (('bright', 'bright'), ('dark', 'dark'), ('tomo_images', 'tomo')):
        t.setdefault(key, {})
        t[key]['filename'] = str(Path(names[role]).resolve())
    t['tomo_images']['num_images'] = scan.npoints
    t['spec'] = {'log': str(Path(spec_log).resolve()), 'scan': int(scan.number),
                 'tomo_dir': str(Path(tomo_dir).resolve())}
    if sample_dir:
        t['spec']['sample_dir'] = str(sample_dir)
    return d


def layer_clipping(bright_h5, v, px, rows_band=(600, 1600)):
    """Fraction of beam-column pixels at the detector ceiling in each layer's detector row (bright field)."""
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(bright_h5) as f:
        d = f['entry/data/data']
        nrows = d.shape[1]
        b = np.stack([d[i][rows_band[0]:rows_band[1], 400:2800] for i in range(d.shape[0])])
    frac = (b >= SAT).mean((0, 2))
    rows = np.round(nrows / 2.0 - v / px).astype(int)
    out = np.full(len(v), np.nan)
    ok = (rows >= rows_band[0]) & (rows < rows_band[1])
    out[ok] = frac[rows[ok] - rows_band[0]]
    return out, rows


def suspect_layers(fill, chg, clip):
    """Flag layers whose fill dips below the local trend, or that jump while their detector row is clipped."""
    trend = ndi.median_filter(fill, size=11, mode='nearest')
    dip = (trend - fill) > 0.04
    jump = np.nan_to_num(chg) > 8
    jump_next = np.r_[jump[1:], False]
    near_clip = np.nan_to_num(clip) > 0.5
    return dip | ((jump | jump_next) & near_clip), trend


def ranges_of(idx):
    idx = sorted(int(i) for i in idx)
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def qc(maskfile, bright_h5, analysis, pixel_mm, run_dir, all_layers=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(maskfile)
    m = d['mask']
    v = np.unique(d['Ys']) if 'Ys' in d.files else d['v_mm']
    if m.shape[0] != 70 and all_layers:
        log(f'note: mask has {m.shape[0]} layers')
    fill = m.mean(axis=(1, 2))
    chg = np.r_[np.nan, [(m[i] != m[i - 1]).mean() * 100 for i in range(1, len(m))]]
    clip, rows = layer_clipping(bright_h5, v, pixel_mm)
    bad, trend = suspect_layers(fill, chg, clip)
    edge = [i for i in (0, 1, len(m) - 2, len(m) - 1)]
    log(f'layers: {len(m)}, v {v[0]:+.3f} to {v[-1]:+.3f} mm; fill median {np.median(fill):.2f}; '
        f'layer-to-layer change median {np.nanmedian(chg):.1f}%, p95 {np.nanpercentile(chg, 95):.1f}%')
    log('layers with >50% of beam pixels clipped in the bright field:',
        ranges_of(np.flatnonzero(np.nan_to_num(clip) > 0.5)) or 'none')
    flagged = [i for i in np.flatnonzero(bad) if i not in edge]
    rng = ranges_of(flagged)
    log('suspect layers (fill dip or jump in a clipped row), edges excluded:', rng or 'none')
    log('beam-edge layers (usually unreliable, left for you to trim):', edge)
    if rng:
        log('suggested:  --interp ' + ','.join(f'{a}-{b}' if a != b else str(a) for a, b in rng)
            + '   (supports = 2 layers each side; check them first)')
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].plot(v, fill, 'o-', ms=3, label='fill')
    ax[0].plot(v, trend, '--', label='local trend')
    ax[0].legend()
    ax[0].set_title('filled fraction')
    ax[1].plot(v, chg, 'o-', ms=3)
    ax[1].axhline(8, c='r', lw=.7)
    ax[1].set_title('% voxels differing from previous layer')
    ax[2].plot(v, 100 * clip, 'o-', ms=3)
    ax[2].axhline(50, c='r', lw=.7)
    ax[2].set_title('% bright-field beam pixels clipped (row of each layer)')
    for a_ in ax:
        a_.set_xlabel('v (mm)')
        a_.grid(alpha=.3)
        for i in flagged:
            a_.axvline(v[i], c='orange', lw=2, alpha=.25)
    fig.tight_layout()
    out = run_dir / f'{analysis}_quality.png'
    fig.savefig(out, dpi=65)
    plt.close(fig)
    with open(run_dir / f'{analysis}_suspect_layers.json', 'w') as f:
        json.dump({'suspect_layers': [int(i) for i in flagged], 'ranges': rng, 'edge_layers': edge}, f)
    log('wrote', out)


def run_pipeline(cfg, options=None):
    """Run the stages in ``options.stages`` for the scan described by the nf_config ``cfg``.

    Returns the path of the final mask. Raises ``PipelineError`` if a stage cannot continue.
    """
    opts = options or PipelineOptions()
    tomo = cfg.tomography
    scan_number = opts.scan if opts.scan is not None else tomo.spec.scan
    spec_log = tomo.spec.log
    if scan_number is None or spec_log is None:
        raise PipelineError('the spec log and scan must be set: tomography: spec: {log, scan} in the yml '
                            '(the scan can also be given with --scan)')
    scan_number = int(scan_number)
    tomo_dir = tomo.spec.tomo_dir or '.'
    analysis = cfg._cfg.get('analysis_name') or f'tomo_scan{scan_number}'
    run_dir = Path(opts.run_dir or Path(tomo_dir) / f'run_scan{scan_number}').resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = [s for s in opts.stages if s]
    bad = set(stages) - set(STAGES)
    if bad:
        raise PipelineError(f'unknown stage(s) {sorted(bad)}; choose from {STAGES}')
    if opts.interp is None and 'interp' in stages:
        stages.remove('interp')
        log('no interp ranges given: skipping the interp stage')
    if opts.layers is not None and {'qc', 'interp'} & set(stages):
        log('layers given: qc/interp need all layers, skipping them')
        stages = [s for s in stages if s not in ('qc', 'interp')]

    scans = parse_spec(spec_log)
    if scan_number not in scans:
        raise PipelineError(f'scan {scan_number} not in {spec_log}')
    scan = scans[scan_number]
    sample_dir = tomo.spec.sample_dir
    names = locate(scan_number, scans, tomo_dir, dark=tomo.dark.filename if _has(cfg, 'dark') else None,
                   bright=tomo.bright.filename if _has(cfg, 'bright') else None,
                   tomo=tomo.tomo_images.filename if _has(cfg, 'tomo_images') else None,
                   dry_run=opts.dry_run, sample_dir=sample_dir)
    step = spec_step(scan)
    log(f'scan {scan_number}: {scan.command} ({scan.npoints} points, {step} deg/trigger)')
    log('dark', names['dark'].name, '| bright', names['bright'].name, '| tomo', names['tomo'].name, '| run dir', run_dir)
    stem = names['tomo'].stem
    slots_file = run_dir / f'{stem}_missing_slots.txt'
    merged = nf_config.utils.merge_dicts(yaml.safe_load(DEFAULT_CONFIG), cfg._cfg)      # as the run config will be
    voxel_spacing = merged['tomography']['reconstruction']['voxel_spacing']
    maskfile = run_dir / mask_name(analysis, opts.layers, voxel_spacing)
    run_yml = run_dir / 'tomo_config.yml'

    if 'setup' in stages:
        if not opts.dry_run:
            check_dark_bright(names)
            d = run_config(cfg, names, scan, spec_log, tomo_dir, analysis, run_dir, sample_dir)
            nf_config.save([nf_config.nf_root.NFRootConfig(d)], run_yml)
            log('setup done: tomo_config.yml written')
        else:
            log('setup: dry run, nothing written')

    if 'missing' in stages:
        if slots_file.exists() and not opts.force:
            log('missing: using existing', slots_file.name)
        elif not opts.dry_run:
            try:
                found = find_missing_frames(names['tomo'], expected=scan.npoints, deg_per_slot=step or 0.25,
                                            out_prefix=run_dir / stem, plot=False)
                (run_dir / 'find_missing.log').write_text('\n'.join(found.report) + '\n')
            except (FileNotFoundError, KeyError, RuntimeError, OSError) as exc:
                log(f'find_missing_frames failed ({exc}); falling back to det_exp_out from spec.log '
                    '(cannot see drops after the last spec point)')
                np.savetxt(slots_file, dropped_slots(scan), fmt='%d')
            n = len(slots_file.read_text().split())
            log(f'missing: {n} dropped frames -> {slots_file.name}')

    if 'recon' in stages:
        if maskfile.exists() and not opts.force:
            log('recon: using existing', maskfile.name)
        else:
            recon_opts = ReconstructOptions(
                spec_log=str(Path(spec_log).resolve()), spec_scan=scan_number, missing_slots=str(slots_file),
                max_angle=180, vert_beam_center=opts.vert_beam_center, processes=opts.processes,
                all_layers=opts.layers is None, layer_indices=opts.layers, output_dir=str(run_dir),
                powder_open_um=opts.powder_open_um if opts.powder_open_um > 0 else 0.0)
            if opts.dry_run:
                log('would reconstruct with', recon_opts)
            else:
                run_cfg = nf_config.open_file(str(run_yml))[0]

                def progress(done, total):
                    if done % 10 == 0:
                        log(f'Completed layer {done}/{total}')

                with stage_log(run_dir / 'recon.log'):
                    reconstruct(run_cfg, recon_opts, progress=progress)
                log('recon done ->', maskfile.name)

    if opts.dry_run:
        log('dry run: nothing else executed')
        return maskfile
    if 'qc' in stages:
        pixel_mm = opts.pixel_mm
        if pixel_mm is None:
            panel = next(iter(load_instrument(cfg.input_files.detector_file).detectors.values()))
            pixel_mm = panel.pixel_size_row
        qc(maskfile, names['bright'], analysis, pixel_mm, run_dir, all_layers=opts.layers is None)
    cur = maskfile
    interp_file = maskfile.with_name(maskfile.stem + '_interp.npz')
    if 'interp' in stages:
        if interp_file.exists() and not opts.force:
            log('interp: using existing', interp_file.name)
        else:
            interpolate_mask_file(maskfile, interp_file, parse_ranges(opts.interp))
        cur = interp_file
    elif interp_file.exists() and 'smooth' in stages:
        cur = interp_file
        log('smooth: using existing interpolated mask', cur.name)
    if 'smooth' in stages and opts.smooth_sigma > 0:
        dst = cur.with_name(cur.stem + '_smooth.npz')
        if dst.exists() and not opts.force:
            log('smooth: using existing', dst.name)
        else:
            smooth_mask_file(cur, dst, opts.smooth_sigma)
        cur = dst
    if Path(cur).exists():
        log('final mask:', cur)
    else:
        log('no mask yet: the recon stage has not been run in', run_dir)
    return cur


def _has(cfg, role):
    """True if the yml itself sets tomography: <role>: filename (so it overrides the spec log)."""
    return bool((cfg._cfg.get('tomography') or {}).get(role, {}) and
                (cfg._cfg['tomography'][role] or {}).get('filename'))
