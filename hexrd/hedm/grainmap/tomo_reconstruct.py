"""Reconstruct a tomography mask from a flyscan (``hexrd tomo reconstruct``).

Reads an nf_config yml and the detector file, builds the flat-field-corrected filtered radiograph
stack, runs filtered back-projection per layer in a worker pool, then thresholds, cleans and rebins
each layer to the voxel grid. ``reconstruct`` writes ``<output_directory>/<analysis_name>_tomo_mask.npz``
(``_layer<N>`` or ``_<n>layers`` is added when layers are selected) with the keys ``mask`` (bool,
layers by Ny by Nx), ``Xs``, ``Ys``, ``Zs``, ``voxel_spacing``, ``layer_indices``, ``v_mm`` and ``ramsz`` (the vertical sample
stage position of the scan in mm, nan if unknown, with ``ramsz_source``).

Values that describe the scan (spec log and scan, missing-slots file, angle file, degrees per frame)
come from the ``tomography`` section of the yml and are overridden by the same-named
``ReconstructOptions`` fields (the command-line flags). The number of processes defaults to
``multiprocessing: num_cpus`` of the yml.

The pool functions ``init_reconstruct_worker`` and ``reconstruct_mask_for_layer`` and the global
``_WORKER_CONFIG`` are at module level so that worker processes can import them.
"""
import gc
import logging
import multiprocessing as mp
import os
import re
from dataclasses import replace

import numpy as np
import yaml

from hexrd import instrument
from hexrd.hedm.grainmap import nfutil
from hexrd.hedm.grainmap import tomoutil
from hexrd.hedm.grainmap.tomo_options import ReconstructOptions  # noqa: F401  (re-exported)
from hexrd.hedm.grainmap.tomo_spec import ramsz_of_scan

logger = logging.getLogger(__name__)
_WORKER_CONFIG = None


def resolve_options(cfg, options=None):
    """Options with the unset scan description and process count filled in from the yml."""
    opts = replace(options) if options is not None else ReconstructOptions()
    tomo = cfg.tomography
    if opts.spec_log is None and opts.spec_scan is None:
        opts.spec_log, opts.spec_scan = tomo.spec.log, tomo.spec.scan
    for name in ('missing_slots', 'theta_file', 'deg_per_frame'):
        if getattr(opts, name) is None:
            setattr(opts, name, getattr(tomo, name))
    if (opts.spec_log is None) != (opts.spec_scan is None):
        raise ValueError('the spec log and the spec scan must be given together')
    if opts.processes is None:
        opts.processes = cfg.multiprocessing.num_cpus
    return opts


def tomo_ramsz(cfg, opts):
    """(vertical sample stage position of the tomography scan in mm, where it came from).

    From the spec log if it records the position for the scan; else ``tomography: reconstruction:
    vertical_motor_position`` of the yml; else (nan, 'unknown'). It is saved in the mask so that the NF
    reconstruction can match its layers to the mask (see ``nfutil.resolve_vertical_motor_position``).
    """
    if opts.spec_log is not None:
        ramsz = ramsz_of_scan(opts.spec_log, opts.spec_scan)
        if ramsz is not None:
            return ramsz, 'spec log'
    configured = cfg.tomography.reconstruction.vertical_motor_position
    if configured is not None:
        return float(configured), 'config'
    return float('nan'), 'unknown'


def create_memmap_array(arr, path):
    mm = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    del mm


def init_reconstruct_worker(config):
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)

    _WORKER_CONFIG["filtered_rad_stack"] = np.memmap(
        config["filtered_rad_stack_path"],
        dtype=np.dtype(config["filtered_rad_stack_dtype"]),
        mode='r',
        shape=tuple(config["filtered_rad_stack_shape"])
    )


def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)


def reconstruct_mask_for_layer(task):
    global _WORKER_CONFIG
    cfg = _WORKER_CONFIG

    ii, layer_row = task

    reconstruction_fbp = tomoutil.tomo_reconstruct_layer(
        cfg["filtered_rad_stack"],
        cross_sectional_dim=cfg["cross_sectional_dim"],
        layer_row=layer_row,
        start_tomo_ang=cfg["start_tomo_ang"],
        end_tomo_ang=cfg["end_tomo_ang"],
        y_center=cfg["rot_axis_pos"],
        pixel_size=cfg["pixel_size"],
        bad_frames=cfg["bad_frames"],
        theta=cfg["theta"]
    )

    binary_recon = tomoutil.threshold_and_clean_tomo_layer(
        reconstruction_fbp,
        cfg["recon_thresh"] * 2,
        cfg["noise_obj_size"],
        cfg["min_hole_size"],
        erosion_iter=cfg["erosion_iter"],
        dilation_iter=cfg["dilation_iter"],
        powder_open_um=cfg["powder_open_um"],
        pixel_size_um=cfg["pixel_size"] * 1000.0,
        powder_restore_um=cfg["powder_restore_um"],
        powder_min_object_um2=cfg["powder_min_object_um2"],
        powder_fill_holes_um2=cfg["powder_fill_holes_um2"]
    )

    tomo_mask = tomoutil.crop_and_rebin_tomo_layer(
        binary_recon,
        cfg["voxel_spacing"],
        cfg["pixel_size"],
        x_center=cfg["x_center"],
        z_center=cfg["z_center"],
        x_length=cfg["x_length"],
        z_length=cfg["z_length"]
    )

    fbp = (reconstruction_fbp.astype(np.float32)
           if cfg.get("save_fbp") else None)
    return ii, tomo_mask, fbp


def load_missing_slots(path):
    """Read trigger slot indices (one per line) from a missing_slots file."""
    with open(path) as f:
        tokens = f.read().split()
    missing = np.array([int(tok) for tok in tokens], dtype=int)
    if np.any(missing < 0) or len(np.unique(missing)) != len(missing):
        raise ValueError(f'{path}: slots must be unique and non-negative')
    return np.sort(missing)


_PULSE_RE = re.compile(r'rams pulse (start|step):\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)')


def read_spec_angles(log_path, scan):
    """(start_deg, step_deg) of the frames of a spec scan.

    spec records the trigger start ("rams pulse start", the beginning of the
    first exposure bin) and the step between triggers in the scan's #C
    comments. A frame's angle is the middle of its bin, so frame 0 sits at
    start + step/2; this reproduces the scan's ome column.
    """
    found = {}
    in_scan = False
    with open(log_path, errors='replace') as f:
        for line in f:
            if line.startswith('#S'):
                if in_scan:
                    break
                in_scan = line.split()[1:2] == [str(scan)]
            elif in_scan and line.startswith('#C'):
                m = _PULSE_RE.search(line)
                if m:
                    found[m.group(1)] = float(m.group(2).rstrip('.'))
    if not in_scan:
        raise ValueError(f'scan {scan} not found in {log_path}')
    if 'start' not in found or 'step' not in found:
        raise ValueError(
            f'scan {scan} in {log_path} has no "rams pulse start/step" '
            'comments')
    return found['start'] + found['step'] / 2.0, found['step']


def angle_params(tomo, start_ang, end_ang, opts):
    """(start_deg, step_deg) of slot 0 and per trigger.

    Priority: spec comments (spec log), else the config ome range with
    step (end - start) / (num_imgs - 1). deg_per_frame overrides the step.
    """
    if opts.spec_log is not None:
        start_ang, step = read_spec_angles(opts.spec_log, opts.spec_scan)
        logger.info(
            f'Scan {opts.spec_scan} from {opts.spec_log}: first frame at '
            f'{start_ang:.4f} deg, {step:.6f} deg per trigger')
    else:
        step = None
    if opts.deg_per_frame is not None:
        step = opts.deg_per_frame
    if step is None:
        try:
            num_imgs = tomo.tomo_images.num_imgs
        except AttributeError:
            raise ValueError(
                'Cannot determine the angular step: give a spec log and scan, '
                'deg_per_frame, or set tomography.tomo_images.num_images in '
                'the config')
        step = (end_ang - start_ang) / (num_imgs - 1)
    return start_ang, step


def derive_theta(tomo, ome_range_deg, opts, num_frames):
    """Projection angle per retained frame.

    Priority: theta_file, then missing_slots, then HDF5 trigger
    timestamps (handles dropped frames), then None (tomo_reconstruct_layer
    falls back to a uniform spread over the omega range, which is only right
    if no frame is missing).
    """
    start_ang = ome_range_deg[0][0]
    end_ang = ome_range_deg[0][1]

    if opts.theta_file is not None:
        theta = np.load(opts.theta_file).astype(np.float64)
        if theta.shape != (num_frames,):
            raise ValueError(
                f'{opts.theta_file} has {theta.size} angles but the stack '
                f'has {num_frames} frames')
        logger.info(f'Using angles from {opts.theta_file}')
        return theta

    if opts.missing_slots is not None:
        missing = load_missing_slots(opts.missing_slots)
        start_ang, step = angle_params(tomo, start_ang, end_ang, opts)
        # The stack holds every trigger slot except the missing ones
        nslots = num_frames + len(missing)
        if len(missing) and missing[-1] >= nslots:
            raise ValueError(
                f'{opts.missing_slots}: slot {missing[-1]} is outside the '
                f'{nslots} trigger slots implied by {num_frames} frames + '
                f'{len(missing)} missing')
        slot = np.setdiff1d(np.arange(nslots), missing)
        theta = start_ang + step * slot
        logger.info(
            f'{len(missing)} dropped frames from {opts.missing_slots}; '
            f'angles {theta[0]:.3f} to {theta[-1]:.3f} deg in steps of '
            f'{step:.4f} deg per trigger')
        return theta

    if tomo.filetype != 'h5':
        return None

    start_ang, step = angle_params(tomo, start_ang, end_ang, opts)

    slot, missing = tomoutil.frame_slots_from_h5(tomo.tomo_images.filename)
    if len(slot) != num_frames:
        raise ValueError(
            f'{len(slot)} timestamps but {num_frames} frames in the stack')
    theta = start_ang + step * slot
    logger.info(
        f'{len(missing)} dropped frames; angles {theta[0]:.3f} to '
        f'{theta[-1]:.3f} deg in steps of {step:.4f} deg per trigger')
    return theta


def resolve_tiff_ext(folder, stem, num_digits, start_idx):
    for ext in ('.tif', '.tiff'):
        fname = os.path.join(folder, f'{stem}{start_idx:0{num_digits}d}{ext}')
        if os.path.exists(fname):
            return ext
    raise FileNotFoundError(
        f'Could not find TIFF file with either .tif or .tiff in {folder}'
    )


def load_tomo_data(tomo, rows=None):
    """(rad_stack, image_stack, filtered_rad_stack, tomo_num_imgs, frame_sums).

    h5 stacks are streamed frame by frame straight into a float16 filtered
    stack: rad_stack and image_stack are None and frame_sums holds each full
    frame's total counts. TIFF stacks keep the old in-memory path
    (frame_sums None).
    """
    frame_sums = None
    filetype = tomo.filetype
    img_stem = tomo.img_stem
    num_digits = tomo.num_digits

    if filetype == 'tiff':
        tomo_data_folder = tomo.tomo_images.folder
        tomo_img_start = tomo.tomo_images.img_start
        ext = resolve_tiff_ext(tomo_data_folder, img_stem,
                               num_digits, tomo_img_start)

        tbf_data_folder = tomo.bright.folder
        tbf_img_start = tomo.bright.img_start
        tbf_num_imgs = tomo.bright.num_imgs

        tdf_data_folder = tomo.dark.folder
        tdf_img_start = tomo.dark.img_start
        tdf_num_imgs = tomo.dark.num_imgs

        tomo_num_imgs = tomo.tomo_images.num_imgs

        tdf = tomoutil.gen_median_image(
            tdf_data_folder, tdf_img_start, tdf_num_imgs,
            stem=img_stem, num_digits=num_digits, ext=ext
        )

        tbf = tomoutil.gen_median_image(
            tbf_data_folder, tbf_img_start, tbf_num_imgs,
            stem=img_stem, num_digits=num_digits, ext=ext
        )

        rad_stack, image_stack, filtered_rad_stack = tomoutil.gen_attenuation_rads(
            tomo_data_folder, tbf, tomo_img_start, tomo_num_imgs,
            stem=img_stem, num_digits=num_digits, tdf=tdf, ext=ext
        )

    elif filetype == 'h5':
        tbf_filename = tomo.bright.filename
        tdf_filename = tomo.dark.filename
        tomo_filename = tomo.tomo_images.filename

        tdf = tomoutil.gen_median_image_h5(tdf_filename, rows=rows)
        tbf = tomoutil.gen_median_image_h5(tbf_filename, rows=rows)
        rad_stack = image_stack = None
        filtered_rad_stack, frame_sums = tomoutil.gen_filtered_rads_h5(
            tomo_filename, tbf, tdf, rows=rows
        )
        tomo_num_imgs = len(filtered_rad_stack)

    else:
        raise ValueError(f'Unsupported tomography filetype: {filetype}')

    return rad_stack, image_stack, filtered_rad_stack, tomo_num_imgs, frame_sums


def reconstruct(cfg, options=None, progress=None):
    """Reconstruct the mask described by the nf_config ``cfg`` and return the path of the npz written.

    ``options`` is a ``ReconstructOptions``; what it leaves unset comes from the yml. ``progress``, if
    given, is called as ``progress(layers_done, layers_total)`` after each layer.
    """
    opts = resolve_options(cfg, options)

    output_stem = cfg.analysis_name
    output_dir = opts.output_dir or cfg.output_directory
    det_file = cfg.input_files.detector_file
    tomo = cfg.tomography

    ome_range_deg = tomo.ome_range

    recon_thresh = tomo.processing.recon_thresh
    noise_obj_size = tomo.processing.noise_obj_size
    min_hole_size = tomo.processing.min_hole_size
    erosion_iter = tomo.processing.erosion_iter
    dilation_iter = tomo.processing.dilation_iter

    x_center = tomo.reconstruction.x_center
    z_center = tomo.reconstruction.z_center
    x_length = tomo.reconstruction.x_length
    z_length = tomo.reconstruction.z_length
    voxel_spacing = tomo.reconstruction.voxel_spacing
    v_bnds = tomo.reconstruction.v_bnds

    instr = load_instrument(det_file)
    panel = next(iter(instr.detectors.values()))

    nrows = panel.rows

    pixel_size = panel.pixel_size_row
    rot_axis_pos = panel.tvec[0]
    vert_beam_center = (panel.tvec[1] if opts.vert_beam_center is None
                        else opts.vert_beam_center)

    vert_points = np.arange(
        v_bnds[0] + voxel_spacing / 2.0, v_bnds[1], voxel_spacing)
    center_layer_row = nrows / 2.0 + vert_beam_center / pixel_size
    rows_to_recon = np.round(
        center_layer_row - vert_points / pixel_size).astype(int)

    # Reconstruction circle (diameter, mm) that contains the whole x/z grid.
    # The grid is a rectangle centered at (x_center, z_center) about the
    # rotation axis; its farthest corner sets the radius.
    if opts.cross_sectional_dim is not None:
        cross_sectional_dim = opts.cross_sectional_dim
    else:
        cross_sectional_dim = 2.0 * np.hypot(
            abs(x_center) + x_length / 2.0, abs(z_center) + z_length / 2.0)
    # tomo_reconstruct_layer pads the circle by 1.1; the sinogram has
    # (ncols - 2 * |axis offset|) usable columns
    axis_offset_px = abs(int(np.round(rot_axis_pos / pixel_size)))
    usable_mm = (panel.cols - 2 * axis_offset_px) * pixel_size
    logger.info(
        f'Reconstruction circle {cross_sectional_dim:.3f} mm (x {x_length} x '
        f'z {z_length} mm grid); detector allows {usable_mm / 1.1:.3f} mm')
    if cross_sectional_dim * 1.1 > usable_mm:
        logger.warning(
            f'Grid needs a {cross_sectional_dim:.3f} mm circle (x1.1 padding = '
            f'{cross_sectional_dim * 1.1:.3f} mm) but the detector only '
            f'covers {usable_mm:.3f} mm at this rotation axis; the corners '
            'of the grid will be missing data')

    n_layers = len(vert_points)
    selected = None   # None: every layer in v_bnds
    if opts.layer_indices is not None:
        selected = list(opts.layer_indices)
    elif not opts.all_layers:
        selected = [n_layers // 2 if opts.layer_index is None
                    else opts.layer_index]
    if selected is not None:
        bad_idx = [i for i in selected if not 0 <= i < n_layers]
        if bad_idx:
            raise ValueError(
                f'layer indices {bad_idx} outside 0..{n_layers - 1}')
        vert_points = vert_points[selected]
        rows_to_recon = rows_to_recon[selected]
        output_stem = (f'{output_stem}_layer{selected[0]}'
                       if len(selected) == 1
                       else f'{output_stem}_{len(selected)}layers')
        logger.info(
            f'Layers {selected} of {n_layers}: v = '
            f'{np.round(vert_points, 4).tolist()} mm -> detector rows '
            f'{rows_to_recon.tolist()} (beam center row '
            f'{center_layer_row:.1f})')
    layer_ids = list(range(n_layers)) if selected is None else selected

    # Only the detector rows the layers need (h5 only); rows_to_recon then
    # index into this band
    band = None
    row0 = 0
    if tomo.filetype == 'h5':
        row0 = max(int(rows_to_recon.min()) - opts.row_margin, 0)
        row1 = min(int(rows_to_recon.max()) + opts.row_margin + 1, nrows)
        band = (row0, row1)
        logger.info(f'Loading detector rows {row0}:{row1} of {nrows}')

    (rad_stack, image_stack, filtered_rad_stack, tomo_num_imgs,
     frame_sums) = load_tomo_data(tomo, rows=band)

    theta = derive_theta(tomo, ome_range_deg, opts, tomo_num_imgs)

    # bad frames are judged on the whole frame, not just the band
    frame_intensity = (frame_sums if frame_sums is not None
                       else np.sum(image_stack, axis=(1, 2)))
    bad_frames = np.argwhere(
        frame_intensity < np.median(frame_intensity) * 0.95).ravel()
    if theta is not None and opts.max_angle is not None:
        over = np.flatnonzero(theta > opts.max_angle)
        if len(over):
            logger.info(f'Dropping {len(over)} frames beyond '
                        f'{opts.max_angle} deg: {over.tolist()}')
        bad_frames = np.unique(np.concatenate([bad_frames, over]))
    del rad_stack, image_stack

    filtered_rad_stack = filtered_rad_stack.astype(np.float16, copy=False)

    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(
        output_dir, f'{output_stem}_filtered_rad_stack.dat')

    create_memmap_array(filtered_rad_stack, memmap_path)

    worker_config = {
        "filtered_rad_stack_path": memmap_path,
        "filtered_rad_stack_shape": filtered_rad_stack.shape,
        "filtered_rad_stack_dtype": str(filtered_rad_stack.dtype),
        "start_tomo_ang": ome_range_deg[0][0],
        "end_tomo_ang": ome_range_deg[0][1],
        "tomo_num_imgs": tomo_num_imgs,
        "rot_axis_pos": rot_axis_pos,
        "pixel_size": pixel_size,
        "cross_sectional_dim": cross_sectional_dim,
        "save_fbp": opts.save_fbp,
        "powder_open_um": opts.powder_open_um,
        "powder_restore_um": opts.powder_restore_um,
        "powder_min_object_um2": opts.powder_min_object_um2,
        "powder_fill_holes_um2": opts.powder_fill_holes_um2,
        "bad_frames": bad_frames,
        "theta": theta,
        "recon_thresh": recon_thresh,
        "noise_obj_size": noise_obj_size,
        "min_hole_size": min_hole_size,
        "erosion_iter": erosion_iter,
        "dilation_iter": dilation_iter,
        "voxel_spacing": voxel_spacing,
        "x_center": x_center,
        "z_center": z_center,
        "x_length": x_length,
        "z_length": z_length,
    }

    tasks = list(enumerate(rows_to_recon - row0))
    results = [None] * len(tasks)

    logger.info(f'Starting reconstruction for {len(tasks)} layers')

    try:
        with mp.Pool(
            processes=opts.processes,
            initializer=init_reconstruct_worker,
            initargs=(worker_config,)
        ) as pool:
            for completed, (ii, tomo_mask, fbp) in enumerate(
                pool.imap_unordered(reconstruct_mask_for_layer, tasks), 1
            ):
                results[ii] = tomo_mask
                if fbp is not None:
                    np.save(os.path.join(
                        output_dir,
                        f'{output_stem}_fbp_layer{layer_ids[ii]}.npy'), fbp)
                logger.info(f'Completed layer {completed}/{len(tasks)}')
                if progress is not None:
                    progress(completed, len(tasks))
    finally:
        # the temporary stack is removed even if a worker failed
        gc.collect()
        if os.path.exists(memmap_path):
            os.remove(memmap_path)

    full_mask = np.array(results)

    if selected is None or len(selected) == 1:
        if selected is not None:
            v_bnds = [vert_points[0] - voxel_spacing / 2.0,
                      vert_points[0] + voxel_spacing / 2.0]
        test_crds, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(
            v_bnds, voxel_spacing, x_center, z_center, x_length, z_length
        )
    else:
        # Non-contiguous layers: one grid per layer, stacked
        grids = [
            nfutil.gen_nf_test_grid(
                [vp - voxel_spacing / 2.0, vp + voxel_spacing / 2.0],
                voxel_spacing, x_center, z_center, x_length, z_length)
            for vp in vert_points
        ]
        Xs, Ys, Zs = (np.concatenate([g[k] for g in grids], axis=0)
                      for k in (2, 3, 4))

    output_file = os.path.join(output_dir, f'{output_stem}_tomo_mask.npz')
    ramsz, ramsz_source = tomo_ramsz(cfg, opts)
    if ramsz_source == 'unknown':
        logger.warning(
            'The vertical stage position (ramsz) of the tomography scan is not in the spec log and not in the '
            'config (tomography: reconstruction: vertical_motor_position); the mask will not record it, so the '
            'NF reconstruction cannot match its layers to the mask from the spec log')
    else:
        logger.info(f'Vertical stage position of the tomography scan: ramsz {ramsz:g} mm ({ramsz_source})')

    np.savez_compressed(
        output_file,
        mask=full_mask.astype(bool),
        Xs=Xs,
        Ys=Ys,
        Zs=Zs,
        voxel_spacing=voxel_spacing,
        layer_indices=np.array(layer_ids),
        v_mm=np.asarray(vert_points),
        ramsz=ramsz,
        ramsz_source=ramsz_source
    )

    logger.info(f'Saved tomography mask to: {output_file}')
    return output_file
