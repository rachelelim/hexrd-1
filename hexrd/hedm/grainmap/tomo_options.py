"""Options of the tomography steps, kept apart from the heavy modules so the command line can build its
parser without importing them (see ``hexrd.hedm.cli.tomo``).

``ReconstructOptions`` are the run-time controls of ``tomo_reconstruct.reconstruct`` and
``PipelineOptions`` those of ``tomo_pipeline.run_pipeline``. ``None`` means "use the yml, or the default".
"""
from dataclasses import dataclass, field
from typing import Optional

STAGES = ['setup', 'missing', 'recon', 'qc', 'interp', 'smooth']


def voxel_label(voxel_spacing):
    """The voxel size in micrometres for a file name: 0.01 mm -> '10um', 0.0045 mm -> '4.5um'."""
    return f'{voxel_spacing * 1000:g}um'


def mask_file_name(stem, voxel_spacing):
    """``<stem>_<step>um_tomo_mask.npz``, the mask file ``tomo_reconstruct.reconstruct`` writes.

    ``stem`` is the analysis name, with ``_layer<N>`` or ``_<n>layers`` if layers were selected.
    """
    return f'{stem}_{voxel_label(voxel_spacing)}_tomo_mask.npz'


@dataclass
class ReconstructOptions:
    """Run-time controls of ``reconstruct``. ``None`` means "use the yml, or the default"."""
    processes: Optional[int] = None         # worker processes (default: multiprocessing:num_cpus)
    theta_file: Optional[str] = None        # .npy, one projection angle (deg) per frame
    missing_slots: Optional[str] = None     # text file, 0-based trigger slots with no frame
    spec_log: Optional[str] = None          # spec.log for the angular step and start angle
    spec_scan: Optional[int] = None         # scan number in spec_log
    deg_per_frame: Optional[float] = None   # angular step per trigger (deg)
    layer_index: Optional[int] = None       # one layer (default: the middle one)
    layer_indices: Optional[list] = None    # exactly these layers
    all_layers: bool = False                # every layer in v_bnds
    save_fbp: bool = False                  # also save each layer's unthresholded reconstruction
    vert_beam_center: Optional[float] = None    # mm, replaces the detector file tvec[1]
    cross_sectional_dim: Optional[float] = None  # mm, diameter of the reconstructed circle
    powder_open_um: float = 0.0             # powder removal: opening radius (um), 0 = off
    powder_restore_um: float = 10.0
    powder_min_object_um2: float = 20000.0
    powder_fill_holes_um2: float = 5000.0
    row_margin: int = 16                    # h5: detector rows loaded around the layers
    max_angle: Optional[float] = None       # drop frames beyond this angle (deg)
    output_dir: Optional[str] = None        # default: output_directory of the yml


@dataclass
class PipelineOptions:
    """Run-time controls of ``run_pipeline``."""
    stages: list = field(default_factory=lambda: list(STAGES))
    force: bool = False
    dry_run: bool = False
    scan: Optional[int] = None              # overrides tomography: spec: scan
    run_dir: Optional[str] = None           # default: <tomo_dir>/run_scan<SCAN>
    layers: Optional[list] = None           # reconstruct only these layers (qc/interp need all)
    interp: Optional[str] = None            # layer ranges to interpolate, e.g. '15-17,28-31,63'
    smooth_sigma: float = 1.5               # voxels (3-D); 0 skips smoothing
    processes: Optional[int] = None         # default: multiprocessing: num_cpus
    vert_beam_center: float = 0.0           # mm; 0 = beam at the detector center
    powder_open_um: float = 40.0            # 0 turns powder removal off
    pixel_mm: Optional[float] = None        # detector pixel size for qc (default: from the detector file)
