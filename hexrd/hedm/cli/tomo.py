"""Command line interface for the tomography mask tools: ``hexrd tomo <step>``.

Steps:
  reconstruct     reconstruct a mask from a flyscan, from an nf_config yml
  pipeline        run every stage (setup, missing, recon, qc, interp, smooth) for one spec scan
  missing-frames  find dropped frames in an h5 stack and write the projection angles
  spec            list the scans in a spec log, or the tomography scans
  interpolate     fill bad layers of a mask by interpolation
  smooth          smooth a mask

The scan description (spec log and scan, detector file, h5 files, grid) is read from the yml, and
the options on the command line control the run and override the yml where both exist. The work is
done by the modules ``tomo_reconstruct``, ``tomo_pipeline``, ``tomo_frames``, ``tomo_spec`` and
``tomo_masks`` in ``hexrd.hedm.grainmap``.
"""
import logging
import os
import sys

_description = 'Tomography masks: reconstruct, pipeline, missing-frames, spec, interpolate, smooth'
_help = 'tomography mask tools (use `hexrd tomo -h` for the steps)'


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('tomo', description=_description, help=_help)
    steps = p.add_subparsers(dest='tomo_step', metavar='step', required=True)
    _configure_reconstruct(steps)
    _configure_pipeline(steps)
    _configure_missing_frames(steps)
    _configure_spec(steps)
    _configure_interpolate(steps)
    _configure_smooth(steps)


def _quiet_flag(p):
    p.add_argument('-q', '--quiet', action='store_true', help='report errors only')


def _setup_logging(args, logfile=None):
    """A console handler (and optionally a log file) on the ``hexrd`` logger, as find-orientations does."""
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.quiet:
        log_level = logging.ERROR
    logger = logging.getLogger('hexrd')
    logger.setLevel(log_level)
    # main() has already put a handler on the root logger (logging.info), which would print every line twice
    was_propagating = logger.propagate
    logger.propagate = False
    handlers = []
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL if args.quiet else log_level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S'))
    handlers.append(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        handlers.append(fh)
    for h in handlers:
        logger.addHandler(h)

    def cleanup():
        logger.propagate = was_propagating
        for h in handlers:
            logger.removeHandler(h)
            h.flush()
            h.close()
    return cleanup


# reconstruct ---------------------------------------------------------------------------------

def _configure_reconstruct(steps):
    from hexrd.hedm.grainmap.tomo_options import ReconstructOptions
    d = ReconstructOptions()
    p = steps.add_parser(
        'reconstruct', description='Reconstruct a tomography mask (one layer, chosen layers, or all)',
        help='reconstruct a mask from a flyscan, from an nf_config yml')
    p.set_defaults(func=execute_reconstruct)
    p.add_argument('yml', nargs='?', default='tomo_config.yml', help='nf_config yml (default tomo_config.yml)')
    _quiet_flag(p)
    p.add_argument('--processes', type=int, default=None,
                   help='worker processes (default: multiprocessing: num_cpus of the yml)')
    p.add_argument('--theta-file', default=None,
                   help='.npy with one projection angle (deg) per frame; overrides angles from the h5 timestamps '
                        '(yml: tomography: theta_file)')
    p.add_argument('--missing-slots', default=None,
                   help='text file with the 0-based trigger slots that have no frame, one per line, as written by '
                        '`hexrd tomo missing-frames`; an empty file means no dropped frames (yml: '
                        'tomography: missing_slots)')
    p.add_argument('--spec-log', default=None,
                   help='spec.log to read the angular step and start angle from; needs --spec-scan '
                        '(yml: tomography: spec: log)')
    p.add_argument('--spec-scan', type=int, default=None,
                   help='scan number of the tomography scan in --spec-log (yml: tomography: spec: scan)')
    p.add_argument('--deg-per-frame', type=float, default=None,
                   help='angular step per trigger (deg); overrides the spec and config value '
                        '(yml: tomography: deg_per_frame)')
    p.add_argument('--layer-index', type=int, default=None,
                   help='reconstruct only this layer (0-based, bottom of v_bnds up); default the middle layer')
    p.add_argument('--layer-indices', type=int, nargs='+', default=None,
                   help='reconstruct exactly these layers (0-based); overrides --layer-index')
    p.add_argument('--all-layers', action='store_true', help='reconstruct every layer in v_bnds')
    p.add_argument('--save-fbp', action='store_true',
                   help="also save each layer's unthresholded reconstruction as <analysis>_fbp_layer<N>.npy")
    p.add_argument('--vert-beam-center', type=float, default=None,
                   help='vertical beam position on the detector (mm), replacing the detector file tvec[1]: the '
                        'height (lab +Y, up) of the detector center above the beam; positive moves the layers '
                        'toward larger row numbers; 0 puts the beam at the center row')
    p.add_argument('--cross-sectional-dim', type=float, default=None,
                   help='diameter (mm) of the circle reconstructed from each sinogram; default the smallest '
                        'circle containing the x/z grid')
    p.add_argument('--powder-open-um', type=float, default=d.powder_open_um,
                   help='powder removal: radius (um) of the opening applied before rebinning; deletes features '
                        'narrower than twice this; 0 (default) is off')
    p.add_argument('--powder-restore-um', type=float, default=d.powder_restore_um,
                   help='with --powder-open-um: keep original pixels within this distance (um) of the opened '
                        'shape (0 = no restoration)')
    p.add_argument('--powder-min-object-um2', type=float, default=d.powder_min_object_um2,
                   help='with --powder-open-um: remove isolated objects smaller than this (um^2)')
    p.add_argument('--powder-fill-holes-um2', type=float, default=d.powder_fill_holes_um2,
                   help='with --powder-open-um: fill holes smaller than this (um^2)')
    p.add_argument('--row-margin', type=int, default=d.row_margin,
                   help='h5 stacks: load the detector rows of the layers plus this many on each side '
                        '(the Gaussian filter reaches 8 rows)')
    p.add_argument('--max-angle', type=float, default=None,
                   help='drop frames whose angle exceeds this value (deg), e.g. 180 when the scan overran')
    p.add_argument('--output-dir', default=None, help='where to write the mask (default: output_directory of the yml)')


def execute_reconstruct(args, parser):
    from hexrd.hedm import nf_config
    from hexrd.hedm.grainmap.tomo_options import ReconstructOptions
    from hexrd.hedm.grainmap.tomo_reconstruct import reconstruct, resolve_options

    cfg = nf_config.open_file(args.yml)[0]
    options = ReconstructOptions(
        processes=args.processes, theta_file=args.theta_file, missing_slots=args.missing_slots,
        spec_log=args.spec_log, spec_scan=args.spec_scan, deg_per_frame=args.deg_per_frame,
        layer_index=args.layer_index, layer_indices=args.layer_indices, all_layers=args.all_layers,
        save_fbp=args.save_fbp, vert_beam_center=args.vert_beam_center,
        cross_sectional_dim=args.cross_sectional_dim, powder_open_um=args.powder_open_um,
        powder_restore_um=args.powder_restore_um, powder_min_object_um2=args.powder_min_object_um2,
        powder_fill_holes_um2=args.powder_fill_holes_um2, row_margin=args.row_margin,
        max_angle=args.max_angle, output_dir=args.output_dir)
    try:
        resolve_options(cfg, options)       # check the options before the long run
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir or cfg.output_directory
    os.makedirs(output_dir, exist_ok=True)
    cleanup = _setup_logging(args, os.path.join(output_dir, f'{cfg.analysis_name}_tomo.log'))
    try:
        reconstruct(cfg, options)
    except (ValueError, FileNotFoundError) as exc:
        logging.getLogger('hexrd').error(str(exc))
        sys.exit(1)
    finally:
        cleanup()


# pipeline ------------------------------------------------------------------------------------

def _configure_pipeline(steps):
    from hexrd.hedm.grainmap.tomo_options import STAGES, PipelineOptions
    d = PipelineOptions()
    p = steps.add_parser(
        'pipeline', description='Run the tomography mask stages for one spec scan: ' + ', '.join(STAGES),
        help='run every stage for one spec scan, from an nf_config yml')
    p.set_defaults(func=execute_pipeline)
    p.add_argument('yml', help='nf_config yml with tomography: spec: {log, scan, tomo_dir}, '
                               'input_files: detector_file and the grid')
    _quiet_flag(p)
    p.add_argument('--scan', type=int, default=None, help='tomography scan number (yml: tomography: spec: scan)')
    p.add_argument('--stages', default=','.join(STAGES), help='comma-separated stages to run (default all)')
    p.add_argument('-f', '--force', action='store_true', help='redo stages whose output exists')
    p.add_argument('--dry-run', action='store_true', help='check the inputs and show what would run')
    p.add_argument('--run-dir', default=None, help='default: <tomo_dir>/run_scan<SCAN>')
    p.add_argument('--layers', type=int, nargs='+', default=None,
                   help='reconstruct only these layers (qc and interp need the full set)')
    p.add_argument('--interp', default=None, help='layer ranges to interpolate, e.g. 15-17,28-31,63')
    p.add_argument('--smooth-sigma', type=float, default=d.smooth_sigma, help='voxels (3-D); 0 skips smoothing')
    p.add_argument('--processes', type=int, default=None,
                   help='worker processes (default: multiprocessing: num_cpus of the yml)')
    p.add_argument('--vert-beam-center', type=float, default=d.vert_beam_center,
                   help='mm; 0 = beam at the detector center')
    p.add_argument('--powder-open-um', type=float, default=d.powder_open_um, help='0 turns powder removal off')
    p.add_argument('--pixel-mm', type=float, default=None,
                   help='detector pixel size (mm) for the qc stage (default: from the detector file)')


def execute_pipeline(args, parser):
    from hexrd.hedm import nf_config
    from hexrd.hedm.grainmap.tomo_options import PipelineOptions
    from hexrd.hedm.grainmap.tomo_pipeline import PipelineError, run_pipeline

    cfg = nf_config.open_file(args.yml)[0]
    options = PipelineOptions(
        stages=[s for s in args.stages.split(',') if s], force=args.force, dry_run=args.dry_run, scan=args.scan,
        run_dir=args.run_dir, layers=args.layers, interp=args.interp, smooth_sigma=args.smooth_sigma,
        processes=args.processes, vert_beam_center=args.vert_beam_center, powder_open_um=args.powder_open_um,
        pixel_mm=args.pixel_mm)
    cleanup = _setup_logging(args)
    try:
        run_pipeline(cfg, options)
    except (PipelineError, ValueError, FileNotFoundError) as exc:
        logging.getLogger('hexrd').error(str(exc))
        sys.exit(1)
    finally:
        cleanup()


# missing-frames ------------------------------------------------------------------------------

def _configure_missing_frames(steps):
    p = steps.add_parser(
        'missing-frames', description='Find dropped frames in an areaDetector h5 stack from its per-frame '
        'timestamps and write the projection angles (see hexrd.hedm.grainmap.tomo_frames)',
        help='find dropped frames in an h5 stack and write the projection angles')
    p.set_defaults(func=execute_missing_frames)
    p.add_argument('h5file')
    p.add_argument('--dataset', default='/entry/data/data')
    p.add_argument('--attr-group', default='/entry/instrument/NDAttributes')
    p.add_argument('--expected', type=int, default=None, help='number of frames the scan should have produced')
    p.add_argument('--deg-per-slot', type=float, default=0.25, help='angular step per trigger (default 0.25)')
    p.add_argument('--span-deg', type=float, default=None,
                   help='force first-to-last retained frame to span this many degrees (overrides --deg-per-slot)')
    p.add_argument('--start-deg', type=float, default=0.0, help='angle of the first retained frame (default 0)')
    p.add_argument('--period', type=float, default=None,
                   help='trigger period in seconds (default: median frame interval)')
    p.add_argument('--tol', type=float, default=0.35,
                   help='allowed deviation of interval/period from an integer (default 0.35)')
    p.add_argument('--out-prefix', default=None, help='default: the h5 path without extension')
    p.add_argument('--no-plot', action='store_true')
    _quiet_flag(p)


def execute_missing_frames(args, parser):
    from hexrd.hedm.grainmap.tomo_frames import find_missing_frames

    cleanup = _setup_logging(args)
    try:
        found = find_missing_frames(
            args.h5file, dataset=args.dataset, attr_group=args.attr_group, expected=args.expected,
            deg_per_slot=args.deg_per_slot, span_deg=args.span_deg, start_deg=args.start_deg,
            period=args.period, tol=args.tol, out_prefix=args.out_prefix, plot=not args.no_plot)
    except (FileNotFoundError, KeyError, RuntimeError, OSError) as exc:
        logging.getLogger('hexrd').error(str(exc))
        sys.exit(2)
    finally:
        cleanup()
    if not args.quiet:
        print('\n'.join(found.report))


# spec ----------------------------------------------------------------------------------------

def _configure_spec(steps):
    p = steps.add_parser('spec', description='List the scans of a spec log (see hexrd.hedm.grainmap.tomo_spec)',
                         help='list the scans in a spec log, or the tomography scans')
    p.set_defaults(func=execute_spec)
    p.add_argument('logfile')
    p.add_argument('scan', nargs='?', type=int, help='show details of this scan')
    p.add_argument('--tomo', action='store_true', help='list tomography scans with det_exp_out statistics')
    p.add_argument('--missing-slots-dir', metavar='DIR',
                   help='write a <retiga stem>_missing_slots.txt (one 0-based slot per line) for every tomo scan, '
                        'from det_exp_out')
    p.add_argument('--csv', metavar='OUT.csv', help='with --tomo: write per-point det_exp_out for every tomo scan')


def execute_spec(args, parser):
    from hexrd.hedm.grainmap import tomo_spec

    scans = tomo_spec.parse_spec(args.logfile)
    if args.missing_slots_dir:
        for number, n, path in tomo_spec.write_missing_slots(scans, args.missing_slots_dir):
            print(f'scan {number:>3}  {n:>3} dropped  -> {path}')
    elif args.tomo:
        print('\n'.join(tomo_spec.tomo_summary(scans)))
        if args.csv:
            tomo_spec.write_det_exp_out_csv(scans, args.csv)
            print(f'wrote {args.csv}')
    elif args.scan is None:
        print(f'{len(scans)} scans in {args.logfile}')
        print('\n'.join(tomo_spec.scan_table(scans)))
    else:
        if args.scan not in scans:
            parser.error(f'scan {args.scan} not in {args.logfile}')
        print('\n'.join(tomo_spec.scan_details(scans[args.scan])))


# interpolate and smooth ----------------------------------------------------------------------

def _configure_interpolate(steps):
    p = steps.add_parser(
        'interpolate', description='Fill bad layers of a tomo mask by cubic interpolation of signed-distance maps',
        help='fill bad layers of a mask by interpolation')
    p.set_defaults(func=execute_interpolate)
    p.add_argument('mask', help='*_tomo_mask.npz')
    p.add_argument('out', help='output npz')
    p.add_argument('ranges', help='layer ranges to fill, e.g. 31-34 or 15-17,28-31,63; each is interpolated '
                                  'from the two layers on each side')
    p.add_argument('--support', type=int, nargs=4, default=None, metavar='LAYER',
                   help='the four layers to interpolate from, for a single range')
    _quiet_flag(p)


def execute_interpolate(args, parser):
    from hexrd.hedm.grainmap.tomo_masks import interpolate_mask_file, parse_ranges

    cleanup = _setup_logging(args)
    try:
        interpolate_mask_file(args.mask, args.out, parse_ranges(args.ranges), args.support)
    except ValueError as exc:
        logging.getLogger('hexrd').error(str(exc))
        sys.exit(1)
    finally:
        cleanup()


def _configure_smooth(steps):
    p = steps.add_parser(
        'smooth', description='Smooth a tomo mask by Gaussian-filtering its signed-distance field',
        help='smooth a mask')
    p.set_defaults(func=execute_smooth)
    p.add_argument('mask', help='*_tomo_mask.npz')
    p.add_argument('out', help='output npz')
    p.add_argument('sigma', type=float, help='filter width in voxels')
    p.add_argument('--2d', dest='per_layer', action='store_true', help='filter each layer on its own')
    _quiet_flag(p)


def execute_smooth(args, parser):
    from hexrd.hedm.grainmap.tomo_masks import smooth_mask_file

    cleanup = _setup_logging(args)
    try:
        smooth_mask_file(args.mask, args.out, args.sigma, per_layer=args.per_layer)
    finally:
        cleanup()
