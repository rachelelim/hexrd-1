#!/usr/bin/env python
"""Tomography mask pipeline for one spec scan.

    python run_tomo_pipeline.py SCAN --spec SPEC.LOG --detector DETECTOR.yml [options]
    (run with the hexrdgui_pr_Oct2026 python; the retiga_*.h5 files are looked up in --tomo-dir, default: cwd)

Stages (run in this order; a stage is skipped if its output exists, unless --force):
  setup   find dark/bright/tomo files from spec.log, write <run-dir>/tomo_config.yml (from tomo_config.yml here)
  missing find dropped frames from the h5 timestamps  -> <run-dir>/<retiga>_missing_slots.txt
  recon   process_tomo_clean.py (spec angles, missing slots, powder removal) -> <name>_tomo_mask.npz
  qc      per-layer report: fill, layer-to-layer change, bright-field clipping; suggests bad layers
  interp  interpolate the layer ranges given with --interp           -> <name>_tomo_mask_interp.npz
  smooth  smooth the (interpolated) mask                              -> ..._smooth.npz

Typical use:
  python run_tomo_pipeline.py 236 --spec spec.log --detector retiga_e7.yml --tomo-dir . --stages setup,missing,recon,qc
  python run_tomo_pipeline.py 236 --spec spec.log --detector retiga_e7.yml --tomo-dir . --interp 15-17,28-31,63 --stages interp,smooth
"""
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np, scipy.ndimage as ndi

HERE = Path(__file__).resolve().parent                    # this folder (hedm_oct2026/tomo)
sys.path.insert(0, str(HERE))
from parse_spec import parse_spec, dropped_slots           # noqa: E402

STAGES = ['setup', 'missing', 'recon', 'qc', 'interp', 'smooth']
SAT = 4090                                                 # 12-bit detector ceiling is 4095


def log(*a):
    print('[pipeline]', *a, flush=True)


def run(cmd, cwd, logfile=None, dry=False, progress=None):
    log('$', ' '.join(str(c) for c in cmd))
    if dry:
        return 0
    f = open(logfile, 'w') if logfile else None
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    n = 0
    for line in p.stdout:
        if f: f.write(line)
        if progress and progress in line:
            n += 1
            if n % 10 == 0: log(line.strip()[-60:])
    p.wait()
    if f: f.close()
    return p.returncode


def mask_name(analysis, layers):
    if layers is None: return f'{analysis}_tomo_mask.npz'
    return f'{analysis}_layer{layers[0]}_tomo_mask.npz' if len(layers) == 1 else f'{analysis}_{len(layers)}layers_tomo_mask.npz'


def spec_step(scan):
    for c in scan.comments:
        m = re.search(r'rams pulse step:\s*([-0-9.eE+]+?)\.?\s*$', c)
        if m: return float(m.group(1))
    return None


def locate(args, scans):
    """(dark, bright, tomo) retiga h5 paths for the tomo scan, checked against the spec log."""
    s = scans[args.scan]
    if not s.is_tomo: log(f'WARNING: scan {args.scan} was not preceded by the "doing the tomography scan" note')
    num = lambda sc: int(re.search(r'retiga_(\d+)', sc.retiga_file).group(1))
    names = {}
    for role, n in (('dark', args.scan - 2), ('bright', args.scan - 1), ('tomo', args.scan)):
        sc = scans.get(n)
        if sc is None or not sc.retiga_file:
            sys.exit(f'scan {n} ({role}) has no retiga file in spec.log')
        names[role] = Path(args.tomo_dir) / (Path(sc.retiga_file).name)
    if not (num(scans[args.scan - 2]) + 1 == num(scans[args.scan - 1]) and num(scans[args.scan - 1]) + 1 == num(s)):
        log('WARNING: dark/bright/tomo retiga numbers are not consecutive; check --dark/--bright overrides')
    for k in ('dark', 'bright'):
        if getattr(args, k): names[k] = Path(getattr(args, k))
    missing = [str(p) for p in names.values() if not p.exists()]
    if missing and not args.dry_run: sys.exit('h5 file(s) not found: ' + ', '.join(missing))
    return names


def check_dark_bright(names):
    import h5py, hdf5plugin  # noqa: F401
    med = {}
    for k in ('dark', 'bright'):
        with h5py.File(names[k]) as f: med[k] = float(np.median(f['entry/data/data'][0]))
    log(f'frame-0 median counts: dark {med["dark"]:.0f}, bright {med["bright"]:.0f}')
    if not med['dark'] < med['bright']:
        sys.exit('dark and bright look swapped (dark median >= bright); use --dark/--bright to override')


def write_config(args, names, scan, run_dir):
    text = Path(args.template_config).read_text()
    fns = re.findall(r'^(\s+filename:\s*)\S+', text, flags=re.M)
    if len(fns) != 3: sys.exit('template config must have exactly 3 "filename:" entries (bright, dark, tomo_images)')
    order = iter([names['bright'], names['dark'], names['tomo']])
    text = re.sub(r'^(\s+filename:\s*)\S+', lambda m: m.group(1) + str(Path(next(order)).resolve()), text, flags=re.M)
    text = re.sub(r'^(analysis_name:\s*)\S+', lambda m: m.group(1) + args.analysis, text, flags=re.M)
    text = re.sub(r'(detector_file:\s*)"[^"]*"', lambda m: m.group(1) + f'"{Path(args.detector).resolve()}"', text)
    text = re.sub(r'(v_bnds:\s*)\[[^\]]*\]', lambda m: m.group(1) + f'[{args.vbnds[0]}, {args.vbnds[1]}]', text)
    text = re.sub(r'(num_images:\s*)\d+', lambda m: m.group(1) + str(scan.npoints), text)
    (run_dir / 'tomo_config.yml').write_text(text)


def layer_clipping(bright_h5, v, px, rows_band=(600, 1600)):
    """Fraction of beam-column pixels at the detector ceiling in each layer's detector row (bright field)."""
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(bright_h5) as f:
        d = f['entry/data/data']; nrows = d.shape[1]
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
    idx = sorted(int(i) for i in idx); out = []
    for i in idx:
        if out and i == out[-1][1] + 1: out[-1][1] = i
        else: out.append([i, i])
    return out


def qc(args, maskfile, names, scans, run_dir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = np.load(maskfile); m = d['mask']; v = np.unique(d['Ys']) if 'Ys' in d.files else d['v_mm']
    if m.shape[0] != 70 and args.layers is None: log(f'note: mask has {m.shape[0]} layers')
    fill = m.mean(axis=(1, 2))
    chg = np.r_[np.nan, [(m[i] != m[i - 1]).mean() * 100 for i in range(1, len(m))]]
    clip, rows = layer_clipping(names['bright'], v, args.pixel_mm)
    bad, trend = suspect_layers(fill, chg, clip)
    edge = [i for i in (0, 1, len(m) - 2, len(m) - 1)]
    log(f'layers: {len(m)}, v {v[0]:+.3f} to {v[-1]:+.3f} mm; fill median {np.median(fill):.2f}; layer-to-layer change median {np.nanmedian(chg):.1f}%, p95 {np.nanpercentile(chg, 95):.1f}%')
    log('layers with >50% of beam pixels clipped in the bright field:', ranges_of(np.flatnonzero(np.nan_to_num(clip) > 0.5)) or 'none')
    flagged = [i for i in np.flatnonzero(bad) if i not in edge]
    rng = ranges_of(flagged)
    log('suspect layers (fill dip or jump in a clipped row), edges excluded:', rng or 'none')
    log('beam-edge layers (usually unreliable, left for you to trim):', edge)
    if rng:
        log('suggested:  --interp ' + ','.join(f'{a}-{b}' if a != b else str(a) for a, b in rng) + '   (supports = 2 layers each side; check them first)')
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].plot(v, fill, 'o-', ms=3, label='fill'); ax[0].plot(v, trend, '--', label='local trend'); ax[0].legend(); ax[0].set_title('filled fraction')
    ax[1].plot(v, chg, 'o-', ms=3); ax[1].axhline(8, c='r', lw=.7); ax[1].set_title('% voxels differing from previous layer')
    ax[2].plot(v, 100 * clip, 'o-', ms=3); ax[2].axhline(50, c='r', lw=.7); ax[2].set_title('% bright-field beam pixels clipped (row of each layer)')
    for a_ in ax:
        a_.set_xlabel('v (mm)'); a_.grid(alpha=.3)
        for i in flagged: a_.axvline(v[i], c='orange', lw=2, alpha=.25)
    fig.tight_layout(); out = run_dir / f'{args.analysis}_quality.png'; fig.savefig(out, dpi=65)
    json.dump({'suspect_layers': [int(i) for i in flagged], 'ranges': rng, 'edge_layers': edge}, open(run_dir / f'{args.analysis}_suspect_layers.json', 'w'))
    log('wrote', out)


def parse_ranges(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out.append((int(a), int(b or a)))
    return out


def do_interp(args, src, dst):
    from interp_layers import interpolate
    d = dict(np.load(src)); M = d['mask']; n = M.shape[0]; new = M.copy(); done = []; sup = {}
    rngs = parse_ranges(args.interp); bad = {i for a, b in rngs for i in range(a, b + 1)}
    for a, b in rngs:
        s = [a - 2, a - 1, b + 1, b + 2]
        if s[0] < 0 or s[-1] >= n: sys.exit(f'range {a}-{b}: needs layers {s} but the mask has 0..{n - 1}; trim this range instead')
        if set(s) & bad: sys.exit(f'range {a}-{b}: support layers {s} overlap another interpolated range; merge the ranges')
        t = list(range(a, b + 1)); new[t] = interpolate(M, t, s); done += t; sup[f'{a}-{b}'] = s
        log(f'interpolated layers {t} from {s}: fill {np.round(M[t].mean(axis=(1, 2)), 2).tolist()} -> {np.round(new[t].mean(axis=(1, 2)), 2).tolist()}')
    d['mask'] = new; d['interpolated_layers'] = np.array(done); d['interpolation_support'] = np.array(json.dumps(sup))
    np.savez_compressed(dst, **d); log('wrote', dst)


def do_smooth(args, src, dst):
    from smooth_masks import smooth
    d = dict(np.load(src)); m = d['mask']
    new, s = smooth(m, args.smooth_sigma)
    d.update(mask_unsmoothed=m, mask=new, sdf_smooth=s, smooth_sigma_voxels=np.array(args.smooth_sigma))
    np.savez_compressed(dst, **d)
    log(f'wrote {dst}: sigma {args.smooth_sigma} voxels; volume {100 * (new.sum() - m.sum()) / m.sum():+.2f}%, voxels changed {100 * (new != m).mean():.2f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scan', type=int, help='spec scan number of the tomography scan (e.g. 23)')
    ap.add_argument('--spec', required=True, help='spec.log')
    ap.add_argument('--tomo-dir', default='.', help='folder with the retiga_*.h5 files (default: cwd)')
    ap.add_argument('--run-dir', default=None, help='default: <tomo-dir>/run_scan<SCAN>')
    ap.add_argument('--analysis', default=None, help='output name stem (default diamond-1a_scan<SCAN>)')
    ap.add_argument('--template-dir', default=str(HERE), help='folder with process_tomo_clean.py, find_missing_frames.py, tomo_config.yml')
    ap.add_argument('--template-config', default=None)
    ap.add_argument('--detector', required=True, help='detector yml, e.g. retiga_e7.yml')
    ap.add_argument('--dark'); ap.add_argument('--bright')
    ap.add_argument('--vbnds', type=float, nargs=2, default=[-0.35, 0.35])
    ap.add_argument('--vert-beam-center', type=float, default=0.0, help='mm; 0 = beam at detector center')
    ap.add_argument('--layers', type=int, nargs='+', default=None, help='reconstruct only these layers (qc/interp/smooth need the full set)')
    ap.add_argument('--powder-open-um', type=float, default=40.0, help='0 turns powder removal off')
    ap.add_argument('--processes', type=int, default=8)
    ap.add_argument('--pixel-mm', type=float, default=0.0009)
    ap.add_argument('--interp', default=None, help='layer ranges to interpolate, e.g. 15-17,28-31,63')
    ap.add_argument('--smooth-sigma', type=float, default=1.5, help='voxels (3-D); 0 skips smoothing')
    ap.add_argument('--stages', default='setup,missing,recon,qc,interp,smooth')
    ap.add_argument('--force', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.analysis = args.analysis or f'diamond-1a_scan{args.scan}'
    run_dir = Path(args.run_dir or Path(args.tomo_dir) / f'run_scan{args.scan}').resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    args.template_config = args.template_config or str(Path(args.template_dir) / 'tomo_config.yml')
    stages = [s for s in args.stages.split(',') if s]
    bad = set(stages) - set(STAGES)
    if bad: sys.exit(f'unknown stage(s) {bad}; choose from {STAGES}')
    if args.interp is None and 'interp' in stages:
        stages.remove('interp'); log('no --interp given: skipping the interp stage')
    if args.layers is not None and {'qc', 'interp'} & set(stages):
        log('--layers given: qc/interp need all layers, skipping them'); stages = [s for s in stages if s not in ('qc', 'interp')]

    scans = parse_spec(args.spec)
    if args.scan not in scans: sys.exit(f'scan {args.scan} not in {args.spec}')
    scan = scans[args.scan]; names = locate(args, scans)
    step = spec_step(scan)
    log(f'scan {args.scan}: {scan.command} ({scan.npoints} points, {step} deg/trigger)')
    log('dark', names['dark'].name, '| bright', names['bright'].name, '| tomo', names['tomo'].name, '| run dir', run_dir)
    stem = names['tomo'].stem
    slots_file = run_dir / f'{stem}_missing_slots.txt'
    maskfile = run_dir / mask_name(args.analysis, args.layers)

    if 'setup' in stages:
        if not args.dry_run:
            check_dark_bright(names); write_config(args, names, scan, run_dir)
        log('setup done: tomo_config.yml written')

    if 'missing' in stages:
        if slots_file.exists() and not args.force:
            log('missing: using existing', slots_file.name)
        elif not args.dry_run:
            rc = run([sys.executable, Path(args.template_dir).resolve() / 'find_missing_frames.py', names['tomo'], '--expected', scan.npoints, '--deg-per-slot', step or 0.25,
                      '--out-prefix', run_dir / stem, '--no-plot'], run_dir, run_dir / 'find_missing.log')
            if rc == 2 or not slots_file.exists():
                log('find_missing_frames failed; falling back to det_exp_out from spec.log (cannot see drops after the last spec point)')
                np.savetxt(slots_file, dropped_slots(scan), fmt='%d')
            n = len(slots_file.read_text().split()); log(f'missing: {n} dropped frames -> {slots_file.name}')

    if 'recon' in stages:
        if maskfile.exists() and not args.force:
            log('recon: using existing', maskfile.name)
        else:
            cmd = [sys.executable, (Path(args.template_dir) / 'process_tomo_clean.py').resolve(), 'tomo_config.yml', '--spec-log', Path(args.spec).resolve(), '--spec-scan', args.scan,
                   '--missing-slots', slots_file.resolve(), '--max-angle', 180, '--vert-beam-center', args.vert_beam_center,
                   '--processes', args.processes]
            cmd += ['--all-layers'] if args.layers is None else ['--layer-indices'] + [str(i) for i in args.layers]
            if args.powder_open_um > 0: cmd += ['--powder-open-um', args.powder_open_um]
            rc = run(cmd, run_dir, run_dir / 'recon.log', args.dry_run, progress='Completed layer')
            if rc: sys.exit(f'reconstruction failed (exit {rc}); see {run_dir / "recon.log"}')
            log('recon done ->', maskfile.name)

    if args.dry_run:
        log('dry run: nothing else executed'); return
    if 'qc' in stages: qc(args, maskfile, names, scans, run_dir)
    cur = maskfile
    if 'interp' in stages:
        dst = maskfile.with_name(maskfile.stem + '_interp.npz')
        if dst.exists() and not args.force: log('interp: using existing', dst.name)
        else: do_interp(args, maskfile, dst)
        cur = dst
    elif maskfile.with_name(maskfile.stem + '_interp.npz').exists() and 'smooth' in stages:
        cur = maskfile.with_name(maskfile.stem + '_interp.npz'); log('smooth: using existing interpolated mask', cur.name)
    if 'smooth' in stages and args.smooth_sigma > 0:
        dst = cur.with_name(cur.stem + '_smooth.npz')
        if dst.exists() and not args.force: log('smooth: using existing', dst.name)
        else: do_smooth(args, cur, dst)
        cur = dst
    log('final mask:', cur)


if __name__ == '__main__':
    main()
