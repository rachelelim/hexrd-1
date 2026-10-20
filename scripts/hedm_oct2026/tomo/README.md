# Tomography mask pipeline

Turns a CHESS flyscan tomography scan (Retiga detector, `retiga_*.h5`) into a
binary sample mask on a regular voxel grid, for use with NF-HEDM and for
micromechanical simulation. It was developed on an additively manufactured
316L stainless steel Schwarz-diamond lattice scanned at CHESS ID3A (spec scans 23 and 236, 2026-09)
and the default settings are tuned for that kind of sample; see
[Assumptions and limitations](#assumptions-and-limitations) before applying
it to something different.

One command runs the whole chain for a scan:

```
spec.log + dark/bright/tomo .h5  -->  per-layer reconstruction  -->  mask (70 layers x 150 x 150)
                                      powder removal, rebin           -->  interpolate bad layers
                                                                      -->  smooth  -->  mask + sdf
```

## Contents of this folder

| File | Role |
|---|---|
| `run_tomo_pipeline.py` | **Driver.** Runs all stages for one spec scan. Start here. |
| `process_tomo_clean.py` | Reconstruction (FBP per layer), thresholding, powder removal, rebinning to the voxel grid. |
| `tomoutil_REL.py` | Library for the above: streaming loader, attenuation, FBP, cleaning, rebinning. |
| `find_missing_frames.py` | Finds dropped frames from the h5 timestamps and writes the missing-slots file. |
| `parse_spec.py` | Reads `spec.log`: scans, commands, columns (`ome`, `det_exp_out`, ...), comments. Also usable as a CLI. |
| `interp_layers.py` | Fills bad layers by cubic interpolation of signed-distance maps. |
| `smooth_masks.py` | Smooths a mask by filtering its signed-distance field in 3D. |
| `tomo_config.yml` | Template config; the driver rewrites the filenames, `v_bnds`, `num_images` and `analysis_name` per scan. |

## Requirements

Use the `hexrdgui_pr_Oct2026` conda environment (hexrd with `hexrd.hedm.nf_config`).
The scripts also need `h5py`, `hdf5plugin`, `numpy`, `scipy`, `scikit-image`, `PyYAML`, `matplotlib`.

`hdf5plugin` is required: the Retiga files use the HDF5 LZ4 filter (id 32004), and without it
reading fails with `OSError: Can't synchronously read data (can't open directory .../hdf5/plugin)`.
`pip install hdf5plugin` fixes it.

`process_tomo_clean.py` finds the `nfutil` package in the parent (`hedm_oct2026`) folder, so keep the
scripts where they are and run them in place. The driver already does this.

## Inputs

1. **`spec.log`** for the beamtime. For tomography scan `N` the driver expects
   - scan `N`: the tomo scan, `flyscan ome 0 180 720 <exposure>`, preceded in the log by the note
     "doing the tomography scan";
   - scan `N-2`: dark field and scan `N-1`: bright field (both `flyscan ome 0 10 20 <exposure>`, 21 frames).
   Each scan's `Retiga #0 saved last image to .../retiga_XXXXXX.h5` line gives its image file.
   The driver warns if the retiga numbers are not consecutive or the dark looks brighter than the bright;
   override with `--dark` / `--bright`.
2. **The three `retiga_*.h5` files**, all in one folder (`--tomo-dir`). Dataset `entry/data/data`,
   one frame per chunk, with `NDAttributes` timestamps.
3. **A detector yml** (`retiga_e7.yml`): pixel size (0.9 um here) and the rotation-axis column offset
   (`tvec[0]`, 0 here).

## Quick start

Run with the `hexrdgui_pr_Oct2026` Python. Paths below are examples.

```bash
# 1. set up, find dropped frames, reconstruct all 70 layers, write the quality report  (~13 min, ~4 GB RAM)
python run_tomo_pipeline.py 236 --spec /path/spec.log --detector /path/retiga_e7.yml \
    --tomo-dir /path/tomo --stages setup,missing,recon,qc

# 2. look at <run-dir>/<name>_quality.png and the printed suggestions, then
#    interpolate the bad layer ranges and smooth
python run_tomo_pipeline.py 236 --spec /path/spec.log --detector /path/retiga_e7.yml \
    --tomo-dir /path/tomo --interp 15-17,28-31,63 --stages interp,smooth
```

`--dry-run` prints the commands without running them. A stage is skipped if its output exists;
add `--force` to redo it. Everything is written to `--run-dir` (default `<tomo-dir>/run_scan<N>`).
`--layers 52` reconstructs a single layer (`qc` and `interp` need all 70 and are skipped).

## Stages

| Stage | What it does | Output (in the run dir) |
|---|---|---|
| `setup` | Finds dark/bright/tomo files from `spec.log`, checks dark < bright, writes the config. | `tomo_config.yml` |
| `missing` | Runs `find_missing_frames.py` on the tomo h5. Falls back to `det_exp_out` from `spec.log` if that fails. | `<retiga>_missing_slots.txt` (+ `_frame_angles.csv`, `_theta_deg.npy`) |
| `recon` | Runs `process_tomo_clean.py` (details below). | `<name>_tomo_mask.npz`, `recon.log` |
| `qc` | Per-layer fill, layer-to-layer change, bright-field clipping; flags suspect layers. | `<name>_quality.png`, `<name>_suspect_layers.json` |
| `interp` | Interpolates the ranges given by `--interp` (2 supporting layers each side). | `<name>_tomo_mask_interp.npz` |
| `smooth` | Smooths the interpolated (or original) mask; `--smooth-sigma 0` skips it. | `..._smooth.npz` |

`<name>` is `--analysis` (default `diamond-1a_scan<N>`).

## What the reconstruction does

For each requested layer (a detector row), `process_tomo_clean.py`:

1. **Loads only the detector rows it needs**, streamed frame by frame: the band covering the layers plus a
   16-row margin (`--row-margin`; the Gaussian filter reaches 8 rows, so 16 is exact). Dark and bright are
   per-pixel medians of their 21 frames. The result is a float16 filtered stack, so a 70-layer run peaks
   near 4 GB.
2. **Attenuation per frame:** `-log((I - dark) / (bright - dark))`, then a Gaussian filter (sigma 2 px).
3. **Projection angles** from the `spec.log` comments of the tomo scan: `rams pulse step` is the angle step
   per trigger, and frame 0 sits at `rams pulse start + step/2` (frames are bin midpoints; this reproduces the
   scan's `ome` column). Slots in the missing-slots file are skipped, so every kept frame gets the angle of
   the trigger that produced it. Frames beyond `--max-angle` (use 180) are dropped.
4. **Bad frames** (total counts below 95% of the median) are removed from the reconstruction.
5. **Filtered back-projection** of the layer's sinogram (`skimage.transform.iradon`), on a circle whose
   diameter is derived from the x/z grid: `2*hypot(|x_center| + x_length/2, |z_center| + z_length/2)`.
   The code pads it by 1.1. Override with `--cross-sectional-dim`.
6. **Threshold** at `recon_thresh * 2` (config), then a closing (`dilation_iter`, `erosion_iter`).
7. **Powder removal** (if `--powder-open-um > 0`), at full resolution (0.9 um/px), before rebinning:
   - an opening with a disk of radius `--powder-open-um` deletes everything narrower than twice that radius
     (powder particles, necks, thin roughness) and keeps thick lattice walls;
   - `--powder-restore-um` gives back original pixels within that distance of the opened shape, so true wall
     edges survive;
   - objects smaller than `--powder-min-object-um2` are removed and holes smaller than
     `--powder-fill-holes-um2` are filled (these replace the config `noise_obj_size` / `min_hole_size`).
8. **Rebin and crop** to the voxel grid (`voxel_spacing`, `x_length` x `z_length` in the config).

### Geometry and layer indexing

- Layer index 0 is the bottom of `v_bnds`; v increases upward. With the default `v_bnds: [-0.35, 0.35]` and a
  0.01 mm voxel there are 70 layers at v = -0.345, -0.335, ..., +0.345 mm. The default middle layer is index 35.
- **Coordinate convention** (hexrd lab frame): the origin is at the sample center. Looking from the sample at the
  detector, +X is to the right, +Y is up and +Z is upstream (toward the beam source, so the detector sits at
  negative Z). The detector file's `translation` is the position of the detector center in this frame (mm). For
  `retiga_e7.yml` it is `[0, 0.974, -4.59]`.
- Detector row of a layer = `nrows/2 + beam_center/pixel - v/pixel`, with image row 0 at the top (so +Y/+v is
  toward smaller row numbers). `--vert-beam-center` is that `beam_center`, in mm. Under the convention above it equals
  the Y of the detector center: the height of the detector center above the sample origin, positive when the
  detector center is above the beam, which moves the layers down the image. The default (omitted
  `--vert-beam-center`) uses `translation[1]` of the detector file.
- **For scans 23 and 236 the detector file's 0.974 mm does not describe the tomography geometry**: with it, the
  layers land near row 2158, outside the beam, and the mask is speckle. The beam is at the detector center,
  so the driver passes `--vert-beam-center 0`. Check this for each new beamtime or sample height.
- The rotation axis is at column `ncols/2 - translation[0]/pixel` (derived from the code and the +X-right
  convention, not tested with a non-zero offset). It is 0 for `retiga_e7.yml`.
- The mask grid is 150 x 150 voxels covering 1.5 mm (X and Z from -0.75 to +0.74 mm).

## Output file format

`<name>_tomo_mask.npz`:

| Key | Content |
|---|---|
| `mask` | bool, (layers, 150, 150) |
| `Xs`, `Ys`, `Zs` | voxel-center coordinates in mm, same shape as `mask` (`Ys` = v) |
| `voxel_spacing` | mm |
| `layer_indices`, `v_mm` | which layers are in the file and their v |

The `_interp` file adds `interpolated_layers` and `interpolation_support` (JSON: the supporting layers used
for each range). The `_smooth` file adds `mask_unsmoothed`, `sdf_smooth` and `smooth_sigma_voxels`.
Always check `interpolated_layers`: those layers are not measurements.

## Quality control and bad layers

**Detector saturation is the main failure mode.** The Retiga is 12-bit, so counts clip at 4095. Where the
bright field is clipped, the flat-field correction is wrong: the sample looks less absorbing, the threshold
misclassifies it, and rings come out broken. A longer exposure clips more: in the lattice data, the 0.2 s
scan (23) had 4 bad layers and the 0.25 s scan (236) had 8 (the quality check suggested 11).

`qc` prints and plots, per layer: filled fraction, % of voxels that differ from the previous layer, and the %
of bright-field beam pixels at the ceiling in that layer's detector row. It flags a layer as suspect when
its fill falls more than 0.04 below the local trend (median over 11 layers), or it jumps more than 8% from
a neighbor while more than half its row is clipped. It also lists the beam-edge layers (0, 1, 68, 69), which are
usually unreliable. The suggested `--interp` ranges are only suggestions; look at the figure first.
On scans 23 and 236 it found all the known bad layers plus a few borderline neighbors.

Clipping can't be repaired from the data. If a flat field exists that is below saturation (shorter exposure or
filtered beam), use it as the bright field via `--bright`.

## Interpolating bad layers

`--interp 15-17,28-31,63` rebuilds each range from the two good layers on each side (for `28-31`: 26, 27, 32, 33).
For each voxel the signed distance to the mask edge is interpolated with a cubic through those four layers,
then thresholded at 0. Tested on good layers (hide four, rebuild from the neighbors), it gets 1.4 to 3.3% of
voxels wrong, about one normal layer-to-layer change (2.2%). The supports must be good layers and must not overlap
another range; the driver refuses otherwise, and refuses ranges that run off the ends. Trim those instead.

Interpolation can't recover real features shorter than the gap (about 40 um for a four-layer range).

## Smoothing

`--smooth-sigma 1.5` (voxels, so 15 um at 0.01 mm) filters the signed-distance field in 3D and thresholds at 0.
Cost on scans 23 and 236: volume change below 0.2%, about 1.5% of voxels flip, boundary-face count falls about 20%,
the mask stays one connected piece. Widths of 2 voxels or more split scan 236 into extra pieces. A
per-layer 2D filter helps less because it leaves the steps between layers.

A binary mask on a grid keeps a staircase on diagonal edges. For a smooth surface, mesh `sdf_smooth` at 0
(e.g. `skimage.measure.marching_cubes(sdf_smooth, 0)`); it gives about 77,000 vertices for these masks.
Watertightness of the meshes has not been checked.

## Using the scripts individually

```bash
# list tomo scans and det_exp_out statistics from spec.log; dropped-frame files for all of them
python parse_spec.py spec.log --tomo [--csv det_exp_out.csv]
python parse_spec.py spec.log --missing-slots-dir missing_slots

# dropped frames from the h5 (exit status 1 just means "frames are missing")
python find_missing_frames.py retiga_012547.h5 --expected 721 --deg-per-slot 0.249788 --out-prefix out/retiga_012547

# reconstruction only (run from the run folder; config must point at the h5 files)
python /path/to/process_tomo_clean.py tomo_config.yml --spec-log spec.log --spec-scan 23 \
    --missing-slots retiga_012547_missing_slots.txt --max-angle 180 --vert-beam-center 0 \
    --all-layers --powder-open-um 40 --processes 8
#   one layer: --layer-index 52      a list: --layer-indices 3 10 17      keep the raw FBP: --save-fbp

python interp_layers.py mask.npz out.npz 31 34 29 30 35 36     # layers 31-34 from 29, 30, 35, 36
python smooth_masks.py in.npz out_smooth.npz 1.5
```

## Parameters

| Option | Default | Notes |
|---|---|---|
| `--powder-open-um` | 40 in the driver (0 = off in `process_tomo_clean.py`) | Walls here are 100s of um thick; a 40 um opening cost about 4% of clean-layer area, 2.75% with restoration. |
| `--powder-restore-um` | 10 | At 20 um or more, powder starts coming back. |
| `--powder-min-object-um2` | 20000 | Removes all isolated powder in the test layers; any value from 10000 up worked. |
| `--powder-fill-holes-um2` | 5000 | 5000 and 20000 gave nearly identical results. |
| `--smooth-sigma` | 1.5 | voxels |
| `--vbnds` | -0.35 0.35 | mm; matches the 700 um beam height. |
| `recon_thresh` (config) | 0.0002 | The code applies `* 2`. The FBP histogram of this sample has two peaks (background ~0, material ~0.0008) and 0.0004 sits at the minimum between them. |

## Assumptions and limitations

- **Vertical position:** the beam is assumed to be at the detector center (`--vert-beam-center 0`), which is
  what the scans here showed (beam on rows ~700 to 1500, centered on row 1100). The script ignores
  `vertical_motor_position` from the config. For a scan at a different height, such as the other half of a
  structure, set `--vert-beam-center` to the Y of the detector center relative to the beam. The sign follows
  the coordinate convention above and was derived from the code; it has not been tested on a scan with a non-zero
  offset, so check where the layers land on a radiograph.
- **Mask orientation:** the mask's X/Z axes come from the same grid function as the NF reconstruction
  (`nfutil.gen_nf_test_grid`), but its in-plane orientation relative to the lab +X/+Z axes (and the sense of
  omega rotation) has not been checked against an NF reconstruction. Do that before using the mask for NF work.
- **Rotation axis:** taken from the detector file (`translation[0]`). If the sample is remounted, check that the
  sinogram is centered.
- **Powder:** the powder settings treat loose or packed powder as void. This was a deliberate choice for a
  micromechanical simulation. For a sample where such material should count, run without `--powder-open-um`.
  Parameters were tuned on a lattice with thick walls; thin struts below about 80 um would be damaged by a 40 um opening.
- **Detector assumptions** in `qc`: 12-bit ceiling (4095), pixel size 0.0009 mm (`--pixel-mm`), beam columns 400 to 2800,
  rows 600 to 1600. Change them for a different detector or setup.
- **Spec log format:** the pulse start/step comments and the "doing the tomography scan" note come from this beamline's spec macros.
- **Dropped frames:** the missing-slots file lists trigger slots that never produced a frame, found by
  gaps in the h5 timestamps. In scan 23 the h5 held two extra frames after the last programmed angle;
  `--max-angle 180` drops them. The `det_exp_out` fallback can't see drops after the last spec point.
- **Time and memory:** about 13 minutes for 70 layers with 8 processes on a 10-core laptop; peak memory about 4 GB; a temporary
  float16 memmap (about 3.3 GB) is written to the run dir and removed at the end.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OSError: ... can't open directory .../hdf5/plugin` | Install `hdf5plugin` in the environment. |
| `ModuleNotFoundError: nfutil` | Run `process_tomo_clean.py` in place (not a copy elsewhere); it looks for `nfutil` in the parent folder. |
| "dark and bright look swapped" | Check which scans are dark/bright; pass `--dark` / `--bright`. |
| Broken or C-shaped rings in some layers | Bright-field clipping at 4095; see Quality control. |
| Mask is mostly speckle | The layer rows are outside the beam: check `--vert-beam-center` and `v_bnds` against a radiograph. |
| Reconstruction needs far too much memory | Use the streaming path (h5 only); TIFF stacks still load in full. |

## Worked example: lattice scans 23 and 236

- Scan 23: 721 frames expected, 696 in the h5, 28 missing slots; exposure 0.2 s. Layers 31 to 34 broke (clipped bright field) and were interpolated from 29, 30, 35, 36.
- Scan 236: same sample after vertical compression, exposure 0.25 s, no dropped frames. Layers 15 to 17, 28 to 31 and 63 were interpolated.
- Both: powder open 40 um / restore 10 um, smoothing 1.5 voxels, `v_bnds` -0.35 to 0.35 mm.

The driver was checked against hand-run versions of every stage; the results were identical.
