# Tomography masks: `hexrd tomo`

`hexrd tomo` turns a CHESS flyscan tomography scan (Retiga detector, `retiga_*.h5`) into a binary sample mask on
the voxel grid used by NF-HEDM. Like `hexrd find-orientations` and `hexrd fit-grains`, it is an installed command
that takes a yml config. The full guide, with the assumptions and limits of the method, is
`scripts/hedm_oct2026/tomo/README.md` in the repository; this page is a summary.

## Steps

| Command | What it does |
|---|---|
| `hexrd tomo pipeline <yml>` | Runs all stages for one spec scan: setup, missing, recon, qc, interp, smooth. |
| `hexrd tomo reconstruct <yml>` | Reconstruction (filtered back-projection per layer), thresholding, powder removal, rebinning. |
| `hexrd tomo missing-frames <h5>` | Finds dropped frames from the h5 timestamps; writes the projection angles. |
| `hexrd tomo spec <spec.log> [scan]` | Lists the scans in a spec log, or the tomography scans (`--tomo`). |
| `hexrd tomo offset <spec.log> <tomo scan> <NF scan>` | Prints the vertical offset between a tomography mask and an NF layer, from the stage positions in the spec log. |
| `hexrd tomo interpolate <mask> <out> <ranges>` | Fills bad layers by interpolating signed-distance maps. |
| `hexrd tomo smooth <mask> <out> <sigma>` | Smooths a mask by filtering its signed-distance field. |

`hexrd tomo <step> -h` lists the options of a step. `-q` reports errors only and the global `--debug` is more verbose.

## The config file

The yml is in nf_config format (`hexrd.hedm.nf_config`). `tomo_config.yml` in `scripts/hedm_oct2026/tomo/` is a template.

```yaml
analysis_name: my_sample_S0
output_directory: "./"
input_files:
  detector_file: retiga_e7.yml
tomography:
  spec:                         # the scan; the pipeline finds the dark (scan - 2) and bright (scan - 1) from it
    log: /path/to/spec.log
    scan: 23
    sample_dir: /path/to/sample # h5 files are <sample_dir>/<scan>/nf/retiga_*.h5 (or use tomo_dir, a flat folder)
  reconstruction:
    voxel_spacing: 0.01
    v_bnds: [-0.35, 0.35]
    x_length: 1.5
    z_length: 1.5
  processing:
    recon_thresh: 0.0002
multiprocessing:
  num_cpus: half
```

The pipeline finds the three h5 files from a file set in the yml, else in `<sample_dir>/<scan>/nf/` for scans
`scan - 2` (dark), `scan - 1` (bright) and `scan`, else as the files named in the spec log in `tomo_dir`.

Files and scan description go in the yml: the spec log and scan, the h5 files, the detector, the grid and the
processing thresholds. Options that control a run are command-line flags (`--layer-index`, `--powder-open-um`,
`--stages`, ...), and a flag overrides the same value in the yml, for example `--processes` over
`multiprocessing: num_cpus`.

## Example

```bash
hexrd tomo pipeline scan23.yml --stages setup,missing,recon,qc      # reconstruct all layers and report suspect ones
hexrd tomo pipeline scan23.yml --interp 31-34 --stages interp,smooth
hexrd tomo reconstruct <tomo_dir>/run_scan23/tomo_config.yml --layer-index 20  # one layer, from the run's own config
```

The pipeline writes everything to `<tomo_dir>/run_scan<N>` (`--run-dir` to change it), including the config it used.
A stage is skipped if its output exists (`-f` to redo it), and `--dry-run` checks the inputs without running.

## Matching the mask to the NF layers

The mask heights are measured from the tomography scan's beam center, an NF layer's from its own. The NF reconstruction adds
`vertical_motor_position` to the mask heights, and it is the difference of the vertical sample stage positions (`ramsz`,
positive moves the sample up): `ramsz` of the NF scan minus `ramsz` of the tomography scan. `ramsz` is read from the spec log
(a `#C` comment before the scans at each position), and `hexrd tomo reconstruct` saves it in the mask. In the NF config,
give `NF_reconstruction: tomography: spec_log` and `nf_scan` and the offset is computed; if the spec log cannot be used, the
configured `vertical_motor_position` is used as it is. `hexrd tomo offset spec.log 23 100 --mask mask.npz --span -0.025 0.025`
prints the number and the mask layers the NF run would use. The README in `scripts/hedm_oct2026/tomo` has the details.

## Output

`<name>_tomo_mask.npz` holds `mask` (bool, layers by Ny by Nx), `Xs`, `Ys`, `Zs` (voxel centers, mm),
`voxel_spacing`, `layer_indices`, `v_mm` and `ramsz` (the tomography scan's stage position). The interpolated and smoothed files add the keys listed in the
README (`interpolated_layers`, `interpolation_support`, `mask_unsmoothed`, `sdf_smooth`, ...).

## In Python

```python
from hexrd.hedm import nf_config
from hexrd.hedm.grainmap.tomo_options import ReconstructOptions, PipelineOptions
from hexrd.hedm.grainmap.tomo_reconstruct import reconstruct
from hexrd.hedm.grainmap.tomo_pipeline import run_pipeline

cfg = nf_config.open_file('scan23.yml')[0]
mask_file = reconstruct(cfg, ReconstructOptions(layer_index=20, processes=4))
final = run_pipeline(cfg, PipelineOptions(stages=['setup', 'missing', 'recon']))
```

The modules are `tomo_pipeline`, `tomo_reconstruct`, `tomo_frames`, `tomo_spec`, `tomo_masks` and `tomoutil` in
`hexrd.hedm.grainmap`. Importing them has no side effects (no logging set up, no files created).
