"""Synthetic tomography data for the tomo tests: a disk, its radiographs as areaDetector h5 files, a small
detector yml, a spec log and an nf_config yml. No experimental data is needed.

The disk is centered on the rotation axis, so its mask does not depend on the angle convention.
"""
from pathlib import Path

import h5py
import numpy as np
import yaml
from skimage.transform import radon

PIXEL_MM = 0.0009           # detector pixel size, as for the retiga
ROWS, COLS = 80, 96         # a small detector
DISK_RADIUS_PX = 20
MU = 0.01                   # attenuation of the disk per pixel
BRIGHT, DARK = 1000.0, 10.0
PERIOD = 0.25               # trigger period (s)
VOXEL_MM = 0.0045           # 5 detector pixels
GRID_MM = 0.054             # 12 voxels, inside the circle the detector can reconstruct
V_BNDS = [-0.009, 0.009]    # 4 layers


def disk_image():
    yy, xx = np.mgrid[:COLS, :COLS]
    c = (COLS - 1) / 2.0
    return np.where((yy - c) ** 2 + (xx - c) ** 2 <= DISK_RADIUS_PX ** 2, MU, 0.0)


def ideal_mask(shape):
    """The disk as it should come out on a voxel grid of ``shape``."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= (DISK_RADIUS_PX * PIXEL_MM / VOXEL_MM) ** 2


def angles(n_slots):
    """Projection angle of every trigger slot: n_slots over 180 degrees, ending at 180."""
    step = 180.0 / (n_slots - 1)
    return step * np.arange(n_slots), step


def write_h5(path, frames, slots=None, period=PERIOD, t0=1.7e9):
    """areaDetector-style h5: frames (n, rows, cols) and per-frame timestamps from the trigger slots."""
    n = len(frames)
    slots = np.arange(n) if slots is None else np.asarray(slots)
    t = t0 + period * slots
    with h5py.File(path, 'w') as f:
        f.create_dataset('entry/data/data', data=np.asarray(frames, dtype=np.uint16))
        g = f.create_group('entry/instrument/NDAttributes')
        g['NDArrayEpicsTSSec'] = np.floor(t).astype(np.int64)
        g['NDArrayEpicsTSnSec'] = np.round((t - np.floor(t)) * 1e9).astype(np.int64)
        g['NDArrayUniqueId'] = np.arange(n, dtype=np.int64)


def tomo_frames(n_slots, dropped=()):
    """Radiographs of the disk at every slot except ``dropped``; returns (frames, retained slots)."""
    theta, _ = angles(n_slots)
    sino = radon(disk_image(), theta=theta, circle=True)          # (cols, n_slots) line integrals
    slots = np.array([s for s in range(n_slots) if s not in set(dropped)])
    frames = np.empty((len(slots), ROWS, COLS))
    for i, s in enumerate(slots):
        profile = DARK + (BRIGHT - DARK) * np.exp(-sino[:, s])
        frames[i] = profile[None, :]
    return frames, slots


def write_scan_files(folder, n_slots=90, dropped=(), numbers=(1, 2, 3)):
    """dark, bright and tomo h5 files named retiga_<number>.h5; returns their paths."""
    folder = Path(folder)
    dark = folder / f'retiga_{numbers[0]:06d}.h5'
    bright = folder / f'retiga_{numbers[1]:06d}.h5'
    tomo = folder / f'retiga_{numbers[2]:06d}.h5'
    write_h5(dark, np.full((5, ROWS, COLS), DARK))
    write_h5(bright, np.full((5, ROWS, COLS), BRIGHT))
    frames, slots = tomo_frames(n_slots, dropped)
    write_h5(tomo, frames, slots)
    return dark, bright, tomo


def write_instrument(path, tvec=(0.0, 0.0, -2.0)):
    cfg = {
        'beam': {'energy': 61.0, 'vector': {'azimuth': 90.0, 'polar_angle': 90.0}},
        'id': 'tomo_test',
        'oscillation_stage': {'chi': 0.0, 'translation': [0.0, 0.0, 0.0]},
        'detectors': {'det': {
            'pixels': {'rows': ROWS, 'columns': COLS, 'size': [PIXEL_MM, PIXEL_MM]},
            'saturation_level': 4095,
            'transform': {'tilt': [0.0, 0.0, 0.0], 'translation': list(tvec)}}},
    }
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path


def write_spec_log(path, n_slots=90, dropped=(), numbers=(1, 2, 3), tomo_h5_dir='/data', ramsz=None):
    """A spec log with a dark, a bright and a tomography scan; ``dropped`` slots have a low det_exp_out.
    ``ramsz``, if given, is the vertical stage position logged before the scans."""
    _, step = angles(n_slots)
    lines = ['#F spec.log', '']
    if ramsz is not None:      # logged by a macro just before the scans at a stage position
        lines.append(f'#C Wed Sep 30 21:36:05 2026.  line 2 ramxx 6.10352e-06 ramsz {ramsz}.')
    for role, number in zip(('dark', 'bright', 'tomo'), numbers):
        n = n_slots if role == 'tomo' else 5
        if role == 'tomo':
            lines.append('#C doing the tomography scan')
        lines += [f'#S {number}  flyscan ome 0 180 {n - 1} 0.1', '#D today',
                  f'#C rams pulse start: {-step / 2:.8f}', f'#C rams pulse step: {step:.8f}',
                  '#L ome  det_exp_out']
        for k in range(n):
            lines.append(f'{step * k:.6f}  {0.007 if (role == "tomo" and k in set(dropped)) else 0.2}')
        lines.append(f'Retiga #0 saved last image to {tomo_h5_dir}/retiga_{number:06d}.h5.')
        lines.append('')
    Path(path).write_text('\n'.join(lines))
    return path


def write_config(path, detector, bright=None, dark=None, tomo=None, num_images=90, analysis='synthetic',
                 output_dir='.', spec_log=None, spec_scan=None, tomo_dir=None, sample_dir=None, num_cpus=2,
                 v_bnds=None, extra_tomography=None):
    """An nf_config yml for the synthetic scan. Leave the h5 files out for ``hexrd tomo pipeline``, which
    finds them from the spec log."""
    tomography = {
        'img_stem': 'nf_', 'filetype': 'h5', 'num_digits': 6, 'ome_start': 0.0, 'ome_end': 180,
        'reconstruction': {'voxel_spacing': VOXEL_MM, 'v_bnds': list(v_bnds or V_BNDS), 'x_center': 0,
                           'z_center': 0, 'x_length': GRID_MM, 'z_length': GRID_MM},
        'processing': {'recon_thresh': 0.0003, 'noise_obj_size': 300, 'min_hole_size': 500,
                       'erosion_iter': 1, 'dilation_iter': 1},
    }
    if bright is not None:
        tomography['bright'] = {'filename': str(bright)}
        tomography['dark'] = {'filename': str(dark)}
        tomography['tomo_images'] = {'filename': str(tomo), 'num_images': num_images}
    if spec_log is not None:
        tomography['spec'] = {'log': str(spec_log), 'scan': spec_scan}
        if tomo_dir is not None:
            tomography['spec']['tomo_dir'] = str(tomo_dir)
        if sample_dir is not None:
            tomography['spec']['sample_dir'] = str(sample_dir)
    tomography.update(extra_tomography or {})
    cfg = {'analysis_name': analysis, 'output_directory': str(output_dir),
           'input_files': {'detector_file': str(detector)},
           'tomography': tomography,
           'multiprocessing': {'num_cpus': num_cpus}}
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path


def write_sample_dir(root, n_slots=90, dropped=(), scans=(21, 22, 23), file_numbers=(900021, 900022, 900023)):
    """<root>/<scan>/nf/retiga_<number>.h5 for the dark, bright and tomography scans, as on the beamline storage.

    The file numbers differ from the scan numbers (and from the names ``write_spec_log`` puts in the log), so
    the files have to be found by looking in the scan folders. Returns the three paths."""
    root = Path(root)
    stage = root / '_stage'
    stage.mkdir(parents=True)
    staged = write_scan_files(stage, n_slots, dropped, numbers=file_numbers)
    paths = []
    for scan, src in zip(scans, staged):
        folder = root / str(scan) / 'nf'
        folder.mkdir(parents=True)
        paths.append(src.rename(folder / src.name))
    stage.rmdir()
    return paths
