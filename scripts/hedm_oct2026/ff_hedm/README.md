# FF-HEDM multilayer pipeline

These scripts take the far-field grains that hexrd fits in each layer of a multilayer scan and put
them together into one list of grains, then set up the fit of the next state from it.

| File | Run | What it does |
|---|---|---|
| `ff_pipeline.yml` | (the settings) | Every setting of the pipeline, in one file that all of the scripts read. Copy it for a new sample |
| `build_pipeline.py` | once, or for one state | Writes the hexrd config of every later state and the scripts that run the whole pipeline |
| `calc_lparm.py` | once, on the first state | Finds the energy gradient of the beam and the height of the center of the beam, and saves them |
| `stitch_grains.py` | once, on the first state | Puts the layers of the first state together into the master grains list and the files that seed state 1 |
| `recompile_grains.py` | every state after that | Puts the layers of a state back together into its master list and sets up the next state |
| `ff_config.py`, `ff_grains_io.py` | (imported) | Reading the settings, and the file names, writers and calibration functions that the scripts share |
| `FF_Calibration_Checker.py` | separately | Plots to check an instrument calibration against a grains.out. Not part of this pipeline |

## The order of things

```
hexrd fit-grains, state 0, every layer           <FF>/<stem>S0_L<layer>/grains.out   (by hand)
        |
build_pipeline.py                                <stem>S<n>.yml and run_state_<n>.sh for every state,
        |                                        and run_pipeline.sh
calc_lparm.py                                    <stem>beam_calibration.yml, two figures
        |
stitch_grains.py                                 <stem>S000_master.out
        |                                        S1_seed_L<layer>_grains.out, grain_nums_S0_L<layer>.txt
hexrd fit-grains, state 1, from <stem>S1.yml     <stem>S1_L<layer>/grains.out
        |
recompile_grains.py   --state 1                  <stem>S001_master.out
        |                                        S2_seed_L<layer>_grains.out, grain_nums_S1_L<layer>.txt
hexrd fit-grains, state 2, from <stem>S2.yml     and so on, with recompile_grains.py every time
```

`build_pipeline.py` only writes files and scripts, it does not run anything. The scripts that it writes
do the steps below it in the diagram, from `calc_lparm.py` on.

`<stem>` is `analysis_stem`, which is `<sample>_`. `<FF>` is the `ff_dir` setting.

The stitching is only for the first state, because that is when the grains in the layers have to be
found and matched up from nothing. From then on, hexrd is given each layer's grains from the last
state as a starting point (the seed files), so the grain numbers carry through every state and the
recompile only has to put each grain's layers back together.

## File names

| File | Name | Written by |
|---|---|---|
| hexrd output of a layer | `<FF>/<stem>S<n>_L<layer>/grains.out` | hexrd |
| Master grains of state n | `<stem>S<n:03d>_master.out` | stitch (n = 0), recompile |
| Seed grains of a layer for state n + 1 | `S<n+1>_seed_L<layer>_grains.out` | stitch, recompile |
| Which master rows are in a layer | `grain_nums_S<n>_L<layer>.txt` | stitch, recompile |
| hexrd config of state n | `<stem>S<n>.yml` | build_pipeline |
| Beam calibration | `<stem>beam_calibration.yml` | calc_lparm |

With `location` set to `above` or `below` the seed files are `S<n+1>_seed_above_L...` or
`S<n+1>_seed_below_L...`.

The analysis names in the yaml are `<stem>S<n>_L<layer>`, which is how hexrd names the directory that
`grains.out` ends up in. Every script builds the path to a layer's `grains.out` the same way
(`grains_file_path`), so they find what hexrd wrote.

## Running

Use the `hexrd-nf_Oct2026` environment. Copy `ff_pipeline.yml` and set it for the sample (the settings
are explained in the file, and listed below). Relative paths in it are relative to the file, so the
scripts can be run from anywhere.

### The whole pipeline

```bash
python build_pipeline.py ff_pipeline.yml    # writes the configs and scripts into output_dir
output_dir/run_pipeline.sh                  # calc_lparm, stitch, then every state in order
```

`build_pipeline.py` writes, into `output_dir`:
- `<stem>S<n>.yml` for every state in the config: the first layer is the config in `hexrd: initial_yaml`
  with the images and the seed grains of the state, and each later layer is an update to it.
- `run_state_<n>.sh` for every state: `hexrd fit-grains <stem>S<n>.yml`, then `recompile_grains.py` for
  that state.
- `run_pipeline.sh`: `calc_lparm.py` and `stitch_grains.py` for the first state, then each `run_state_<n>.sh`
  in order. It stops at the first step that fails.

It assumes that hexrd has already fit the first state, with the config that you give as
`hexrd: initial_yaml`. The states are in the config as the scan and file number of their first layer, and
the layers follow in order (`states: 1: {first_scan: 92, first_file: 407}`).

The scripts it writes use `python`, so activate the environment first. On a cluster, edit the hexrd line of
the `run_state_<n>.sh` scripts, or set `hexrd: command` and `hexrd: args` in the config (for example
`args: -f` to overwrite an analysis that is already there), and run them as jobs.

### One step or one state at a time

Every script takes the config, and none of them depend on `run_pipeline.sh`:

```bash
python calc_lparm.py ff_pipeline.yml [--show]             # --show also shows the figures
python stitch_grains.py ff_pipeline.yml
python build_pipeline.py ff_pipeline.yml --state 3        # only state 3: its yaml and run_state_3.sh
output_dir/run_state_3.sh                                 # hexrd, then the recompile, of state 3
python recompile_grains.py ff_pipeline.yml --state 3      # only the recompile, once hexrd has fit state 3
```

A state needs the files that the one before it wrote, so the states have to be done in order. To add a
state later, add it to the config and run `build_pipeline.py --state <n>` for it.

## What each step does

### The frames

hexrd gives every grain's position in the lab frame, where y is up. In a grains.out from a layer, y is
measured in the beam, so it is in the frame of that layer. The layers are moved to different heights
in the sample, so the stitch adds each layer's offset to put the grains in the frame of the sample, and
the seed files take it off again, so that hexrd always sees the frame of a layer. The offset of a layer
is `-ramsz`: the motor is below the sample, so at a higher ramsz the layer is lower in the sample.

### `calc_lparm.py`: energy gradient and beam center

The energy of the beam changes with the height in the beam, so the grains are fit with the wrong
energy by an amount that depends on where in the beam they are. All the lattice spacings scale
together, so it is an error in the mean normal strain, the mean of ln V[0,0], ln V[1,1] and ln V[2,2],
that depends only on y.

It takes the good grains (completeness above `good_comp_min`, chi2 below `good_chi2_max`) from every
layer together and fits that mean strain against y. The gradient of a multilayer monochromator is not
linear, so the default is a cubic spline (`gradient_model = 'spline'`, with `n_knots` interior knots).
For a monochromator with a linear gradient use `'polynomial'` with `poly_degree = 1`. Grains more than
`outlier_sigma` standard deviations off the fit are left out of it.

The fit only covers the heights where there are enough grains: starting at the busiest bin, it goes
outwards until a bin of `trim_bin_width` has fewer than `min_grains_per_bin` good grains. The gradient
is held at its value at those ends and not extrapolated. The middle of this range is the beam center,
`beam_center_y`, and its height should be close to the beam height.

It assumes that the sample has the same mean strain at every height, so that the trend with y is the
gradient and not the sample. Pooling the layers is only fair if they agree, which the figure checks.

It writes:
- `<stem>beam_calibration.yml`: the gradient, with its fit range, and the `beam_center_y`.
- `<stem>energy_gradient.png`: the fit, the residuals, where the grains are and the mean residual of each
  layer (the layers should agree), and a cross-validated comparison of spline knots and polynomial
  degrees, to choose the model by.
- `<stem>grain_density.png`: how many grains are found, and how well they fit, at each height. The beam
  is not equally intense everywhere, and the grains where it is weak are the harder ones to fit. This is
  only to look at, nothing uses it.

### `stitch_grains.py`: the first master list

1. Reads each layer's `grains.out`, and takes the energy gradient out of the strains at each grain's y
   in the frame of its layer.
2. Keeps the grains with completeness of at least `low_comp_thresh` and chi2 of at most `high_chi2_thresh`.
3. Adds the layer offsets, to put the layers in one frame.
4. Removes duplicates. A grain that is seen in two layers is a duplicate of the nearest grain if it is
   within `dist` (mm) and `misorientation` (degrees), allowing for the crystal symmetry. The pair is
   averaged if their completenesses are within `completeness_diff` of each other, with the orientation
   averaged as a quaternion, not as an exponential map, and otherwise the better one is kept.
5. Writes the master list and, if `make_layer_grain_lists` is on (it is by default), the seed files and
   grain number lists of every layer. The hexrd config of the next state is made by `build_pipeline.py`.

The strain correction does not change the positions or the duplicates, only the strain columns.

### `recompile_grains.py`: every state after the first

1. Reads the master list of the last state and, for each layer, the grain numbers that were seeded there.
2. Reads the new `grains.out` of each layer, takes out the energy gradient, adds the layer offset, and puts
   each grain's fit into its row of the master list. A grain has one row per layer, and a row is empty for
   a layer the grain was not seeded in.
3. For every grain, compiles one row from its layers. The layers are those within `close_completeness_cutoff`
   of its best completeness and below `high_chi2_thresh`, and they are averaged. If one layer or fewer has
   a chi2 that low, the layer with the lowest chi2 is used instead.
4. Writes the master list for the state and the seed files and grain number lists for the next one.

A grain that was not seen in any layer is left as zeros and counted as disappeared.

### The seed windows

A grain is seeded in a layer if it is inside a window: the volume that the beam illuminates in that
layer, which is `beam_height` around the beam center, and a margin beyond it. The margin takes in
grains from just outside the illuminated volume, which can move into it. It is `seed_margin` beam
heights, `0.5` by default. `location` sets which sides it is on, with y up:

| `location` | The margin is |
|---|---|
| `both` | above and below the illuminated volume |
| `above` | only above it, toward higher y |
| `below` | only below it, toward lower y |

(These used to be `top` and `bottom`. The old note was that it is generally below for tension and above
for compression.) The highest layer has no upper limit and the lowest has no lower limit, so the grains
beyond the layers on the outside are included too, and no grain is left out of every layer. Windows of
neighbouring layers overlap, so a grain can be seeded in several layers, which is what lets the recompile
average them.

The seed grains of a layer are written with their y in the frame of that layer, with the beam center
where `calc_lparm.py` found it.

## Settings

All of the settings are in the one config, so they cannot disagree between the scripts. The ones that you
set for every sample are `sample`, `ff_dir`, `detector_file`, `materials`, `layers` (`ramsz` and
`beam_height`), and, to set up the later states, `hexrd: initial_yaml`, `hexrd: image_fstem` and `states`.
The rest have defaults, which are the values in the commented-out sections of `ff_pipeline.yml`:

| Section | For | Settings |
|---|---|---|
| `seeds` | seed windows | `location`, `margin` |
| `stitch` | `stitch_grains.py` | `low_comp_thresh`, `high_chi2_thresh`, `dist`, `misorientation`, `completeness_diff`, `v_pos_corr`, `make_layer_grain_lists` |
| `recompile` | `recompile_grains.py` | `close_completeness_cutoff`, `high_chi2_thresh`, `low_comp_thresh` |
| `calibration` | `calc_lparm.py` | `apply`, `good_comp_min`, `good_chi2_max`, `model`, `n_knots`, `poly_degree`, `outlier_sigma`, `min_grains_per_bin`, `trim_bin_width`, `density_bin_width` |
| `hexrd` | the hexrd configs and run scripts | `initial_yaml`, `image_fstem`, `command`, `args` |

A setting that is misspelt, or a state that is numbered wrongly, is an error with the name in it.
`calibration: apply: false` runs the pipeline without the energy gradient, with the beam center at 0.

The detector file is a hexrd instrument file. The stitch reads the beam energy from it and the yaml is made
from its detector keys and panels. The yaml has one image entry per detector, and the panels in an entry are
the ones in that detector's group, so composite detectors with any number of panels work.

## Before using it on a new sample

- Copy `ff_pipeline.yml` and set `sample`, `ff_dir`, `output_dir`, `detector_file`, `materials`,
  `layers`, `hexrd` and `states`. The `states` in the example are made-up scan and file numbers.
- Check the `hexrd: initial_yaml`, the hexrd config that the new yamls are made from, and that its working
  directory is where the scripts will run hexrd (`output_dir`).
- Fit the first state with hexrd, then run the pipeline.
- Refit the calibration (run `calc_lparm.py` again) if the monochromator or the beam position changes
  between states, since the same curve is applied to every state.
- Look at the two figures from `calc_lparm.py`. The layers should lie on one curve in the energy gradient
  figure, and the measured beam height (in the printout) should be close to `beam_height`.
- Set `calibration: apply: false` to run without a calibration.

## Tests

The tests are in `scripts/hedm_oct2026/tests` and use synthetic data, so they need no real files. They run in
a few seconds:

```bash
cd scripts/hedm_oct2026/tests
python -m unittest test_ff_grains_io test_ff_config test_ff_pipeline
```

- `test_ff_grains_io.py`: the file names, seed windows, energy gradient fit and correction, grain lists and
  seed files, and the yaml.
- `test_ff_config.py`: reading the config, the defaults, the paths, and the errors for bad settings.
- `test_ff_pipeline.py`: the whole chain on synthetic grains. It runs `calc_lparm`, `stitch_grains`,
  `build_pipeline` and `recompile_grains` in order, with a simulated hexrd fit between the first state and the
  next, and checks that the grains seen in two layers are merged, the gradient is taken out, the files and
  yamls are right, and that recompiling unchanged fits gives back the master list.
