# %% Necessary Dependencies
import argparse
import logging
import os
import re
import sys
import gc
import multiprocessing as mp

import numpy as np
import yaml

# nfutil lives in the parent (hedm_oct2026) folder
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexrd.hedm import nf_config
import nfutil
import tomoutil_REL as tomoutil

from hexrd import instrument


try:
    import matplotlib.pyplot as plt
    matplot = True
except ImportError:
    logging.warning('no matplotlib, debug plotting disabled')
    matplot = False


# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("tomo_processing.log", mode="a"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
_WORKER_CONFIG = None


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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate tomography mask')
    parser.add_argument(
        'input_file',
        nargs='?',
        default='tomo_config.yml',
        help='Input config file'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help='Number of worker processes'
    )
    parser.add_argument(
        '--theta-file',
        default=None,
        help='.npy file with one projection angle (deg) per frame in the '
             'stack; overrides angles derived from the HDF5 timestamps'
    )
    parser.add_argument(
        '--missing-slots',
        default=None,
        help='Text file with the 0-based trigger slots that have no frame, '
             'one per line (the <prefix>_missing_slots.txt written by '
             'find_missing_frames.py). Angles are rebuilt from the slot '
             'numbers; an empty file means no dropped frames. Used instead '
             'of the HDF5 timestamps, so it also works for TIFF stacks. '
             '--theta-file takes priority if both are given'
    )
    parser.add_argument(
        '--spec-log',
        default=None,
        help='spec.log to read the angular step and start angle from (the '
             '"rams pulse start/step" comments of --spec-scan). Requires '
             '--spec-scan'
    )
    parser.add_argument(
        '--spec-scan',
        type=int,
        default=None,
        help='Scan number in --spec-log for the tomography scan'
    )
    parser.add_argument(
        '--deg-per-frame',
        type=float,
        default=None,
        help='Angular step per detector trigger (deg). Overrides the '
             'spec/config value. Default: the spec "rams pulse step" if '
             '--spec-log is given, else (ome_end - ome_start) / '
             '(num_imgs - 1) from the config'
    )
    parser.add_argument(
        '--layer-index',
        type=int,
        default=None,
        help='Reconstruct only this layer (0-based, bottom of v_bnds up). '
             'Default: the middle layer'
    )
    parser.add_argument(
        '--layer-indices',
        type=int,
        nargs='+',
        default=None,
        help='Reconstruct exactly these layers (0-based), e.g. an evenly '
             'spaced set. Overrides --layer-index'
    )
    parser.add_argument(
        '--save-fbp',
        action='store_true',
        help='Also save each layer\'s unthresholded reconstruction as '
             '<analysis>_fbp_layer<N>.npy'
    )
    parser.add_argument(
        '--all-layers',
        action='store_true',
        help='Reconstruct every layer in v_bnds instead of a single layer'
    )
    parser.add_argument(
        '--vert-beam-center',
        type=float,
        default=None,
        help='Vertical beam position on the detector (mm), replacing the '
             'detector file translation[1]. It is the height (lab +Y, up) of '
             'the detector center above the beam: positive moves the layers '
             'down the image (toward larger row numbers). 0 puts the beam at '
             'the center row'
    )
    parser.add_argument(
        '--cross-sectional-dim',
        type=float,
        default=None,
        help='Diameter (mm) of the circle reconstructed from each sinogram. '
             'Default: the smallest circle containing the x/z grid '
             '(x_length, z_length, x_center, z_center)'
    )
    parser.add_argument(
        '--powder-open-um',
        type=float,
        default=0.0,
        help='Powder removal: radius (um) of the opening applied to the '
             'full-resolution binary before rebinning; deletes features '
             'narrower than twice this. 0 (default) = off. 40 is a good '
             'start for this lattice (walls are hundreds of um thick)'
    )
    parser.add_argument(
        '--powder-restore-um',
        type=float,
        default=10.0,
        help='With --powder-open-um: keep original pixels within this '
             'distance (um) of the opened shape, restoring the true wall edge '
             '(0 = no restoration)'
    )
    parser.add_argument(
        '--powder-min-object-um2',
        type=float,
        default=20000.0,
        help='With --powder-open-um: remove isolated objects smaller than '
             'this (um^2); replaces the config noise_obj_size'
    )
    parser.add_argument(
        '--powder-fill-holes-um2',
        type=float,
        default=5000.0,
        help='With --powder-open-um: fill holes smaller than this (um^2); '
             'replaces the config min_hole_size'
    )
    parser.add_argument(
        '--row-margin',
        type=int,
        default=16,
        help='h5 stacks: load only the detector rows needed for the '
             'reconstructed layers plus this many rows on each side (the '
             'Gaussian filter reaches 8 rows, so 16 is exact)'
    )
    parser.add_argument(
        '--max-angle',
        type=float,
        default=None,
        help='Drop frames whose angle exceeds this value (deg), e.g. 180 '
             'to discard redundant frames when the scan overran'
    )
    args = parser.parse_args()
    if (args.spec_log is None) != (args.spec_scan is None):
        parser.error('--spec-log and --spec-scan must be given together')
    return args


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


def angle_params(tomo, start_ang, end_ang, args):
    """(start_deg, step_deg) of slot 0 and per trigger.

    Priority: spec comments (--spec-log), else the config ome range with
    step (end - start) / (num_imgs - 1). --deg-per-frame overrides the step.
    """
    if args.spec_log is not None:
        start_ang, step = read_spec_angles(args.spec_log, args.spec_scan)
        logger.info(
            f'Scan {args.spec_scan} from {args.spec_log}: first frame at '
            f'{start_ang:.4f} deg, {step:.6f} deg per trigger')
    else:
        step = None
    if args.deg_per_frame is not None:
        step = args.deg_per_frame
    if step is None:
        try:
            num_imgs = tomo.tomo_images.num_imgs
        except AttributeError:
            raise ValueError(
                'Cannot determine the angular step: give --spec-log, '
                '--deg-per-frame, or set tomography.tomo_images.num_imgs in '
                'the config')
        step = (end_ang - start_ang) / (num_imgs - 1)
    return start_ang, step


def derive_theta(tomo, ome_range_deg, args, num_frames):
    """Projection angle per retained frame.

    Priority: --theta-file, then --missing-slots, then HDF5 trigger
    timestamps (handles dropped frames), then None (tomo_reconstruct_layer
    falls back to a uniform spread over the omega range, which is only right
    if no frame is missing).
    """
    start_ang = ome_range_deg[0][0]
    end_ang = ome_range_deg[0][1]

    if args.theta_file is not None:
        theta = np.load(args.theta_file).astype(np.float64)
        if theta.shape != (num_frames,):
            raise ValueError(
                f'{args.theta_file} has {theta.size} angles but the stack '
                f'has {num_frames} frames')
        logger.info(f'Using angles from {args.theta_file}')
        return theta

    if args.missing_slots is not None:
        missing = load_missing_slots(args.missing_slots)
        start_ang, step = angle_params(tomo, start_ang, end_ang, args)
        # The stack holds every trigger slot except the missing ones
        nslots = num_frames + len(missing)
        if len(missing) and missing[-1] >= nslots:
            raise ValueError(
                f'{args.missing_slots}: slot {missing[-1]} is outside the '
                f'{nslots} trigger slots implied by {num_frames} frames + '
                f'{len(missing)} missing')
        slot = np.setdiff1d(np.arange(nslots), missing)
        theta = start_ang + step * slot
        logger.info(
            f'{len(missing)} dropped frames from {args.missing_slots}; '
            f'angles {theta[0]:.3f} to {theta[-1]:.3f} deg in steps of '
            f'{step:.4f} deg per trigger')
        return theta

    if tomo.filetype != 'h5':
        return None

    start_ang, step = angle_params(tomo, start_ang, end_ang, args)

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


def main():
    args = parse_args()
    fname = args.input_file

    cfg = nf_config.open_file(fname)[0]

    output_stem = cfg.analysis_name
    output_dir = cfg.output_directory
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
    vert_beam_center = (panel.tvec[1] if args.vert_beam_center is None
                        else args.vert_beam_center)

    vert_points = np.arange(
        v_bnds[0] + voxel_spacing / 2.0, v_bnds[1], voxel_spacing)
    center_layer_row = nrows / 2.0 + vert_beam_center / pixel_size
    rows_to_recon = np.round(
        center_layer_row - vert_points / pixel_size).astype(int)

    # Reconstruction circle (diameter, mm) that contains the whole x/z grid.
    # The grid is a rectangle centered at (x_center, z_center) about the
    # rotation axis; its farthest corner sets the radius.
    if args.cross_sectional_dim is not None:
        cross_sectional_dim = args.cross_sectional_dim
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
    if args.layer_indices is not None:
        selected = list(args.layer_indices)
    elif not args.all_layers:
        selected = [n_layers // 2 if args.layer_index is None
                    else args.layer_index]
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
        row0 = max(int(rows_to_recon.min()) - args.row_margin, 0)
        row1 = min(int(rows_to_recon.max()) + args.row_margin + 1, nrows)
        band = (row0, row1)
        logger.info(f'Loading detector rows {row0}:{row1} of {nrows}')

    (rad_stack, image_stack, filtered_rad_stack, tomo_num_imgs,
     frame_sums) = load_tomo_data(tomo, rows=band)

    theta = derive_theta(tomo, ome_range_deg, args, tomo_num_imgs)

    # bad frames are judged on the whole frame, not just the band
    frame_intensity = (frame_sums if frame_sums is not None
                       else np.sum(image_stack, axis=(1, 2)))
    bad_frames = np.argwhere(
        frame_intensity < np.median(frame_intensity) * 0.95).ravel()
    if theta is not None and args.max_angle is not None:
        over = np.flatnonzero(theta > args.max_angle)
        if len(over):
            logger.info(f'Dropping {len(over)} frames beyond '
                        f'{args.max_angle} deg: {over.tolist()}')
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
        "save_fbp": args.save_fbp,
        "powder_open_um": args.powder_open_um,
        "powder_restore_um": args.powder_restore_um,
        "powder_min_object_um2": args.powder_min_object_um2,
        "powder_fill_holes_um2": args.powder_fill_holes_um2,
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

    logging.info(f'Starting reconstruction for {len(tasks)} layers')

    with mp.Pool(
        processes=args.processes,
        initializer=init_reconstruct_worker,
        initargs=(worker_config,)
    ) as pool:
        for completed, (ii, tomo_mask, fbp) in enumerate(
            pool.imap_unordered(reconstruct_mask_for_layer, tasks), 1
        ):
            results[ii] = tomo_mask
            if fbp is not None:
                os.makedirs(output_dir, exist_ok=True)
                np.save(os.path.join(
                    output_dir, f'{output_stem}_fbp_layer{layer_ids[ii]}.npy'),
                    fbp)
            logging.info(f'Completed layer {completed}/{len(tasks)}')

    full_mask = np.array(results)

    gc.collect()
    os.remove(memmap_path)

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

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{output_stem}_tomo_mask.npz')

    np.savez_compressed(
        output_file,
        mask=full_mask.astype(bool),
        Xs=Xs,
        Ys=Ys,
        Zs=Zs,
        voxel_spacing=voxel_spacing,
        layer_indices=np.array(layer_ids),
        v_mm=np.asarray(vert_points)
    )

    logging.info(f'Saved tomography mask to: {output_file}')


if __name__ == "__main__":
    mp.freeze_support()
    main()
