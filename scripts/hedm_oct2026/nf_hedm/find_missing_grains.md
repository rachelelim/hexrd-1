# Finding missing grains in NF-HEDM

`find_missing_grains.py` looks for grains that are in the sample but missing from the FF grain
list. They show up in an NF reconstruction as regions of low confidence, because no listed grain
explains the diffraction there. The script searches orientation space at those voxels, keeps the
orientations that fit, and writes them out as extra grains.

## Confidence

Every NF reconstruction step scores an orientation at a voxel the same way. The orientation is used
to simulate its Bragg peaks, and the **confidence** is the fraction of those peaks that land on lit
pixels in the binarized images. A grain that is really at the voxel scores high (about 0.7 to 0.95
here). Random orientations score about 0.1.

The score falls off quickly with misorientation. On Ti-7Al, an orientation 0.5 degrees from the true
one keeps 41% of the true score, and one degree away keeps 23%. This sets how fine the search has to be.

## How the search works

`nfutil.search_orientations` finds the best orientation at each of a batch of voxels, using no
candidate list:

1. **Grid.** Sample the fundamental zone of the crystal symmetry on a grid (1.75 degrees by default,
   about 300,000 orientations for hexagonal).
2. **Score the grid.** Score every grid orientation at every voxel, using the 12 lowest-angle ring
   families. Keep the best 200 grid points per voxel.
3. **Cluster.** Group those points that are closer together than the grid spacing (symmetry aware)
   and keep the best 20 groups. This is the step that matters most: the right basin is often not the
   single highest-scoring grid point.
4. **Coarse refinement.** Refine each group with a pattern search (radii 1.5, 1.0, 0.5 degrees).
   At each radius it scores a 3x3x3 grid of small rotations around the current best, moves to the best
   one, and repeats while the score improves. Keep the best 4.
5. **Fine refinement.** Refine those with radii 0.5, 0.25, 0.1 degrees, now using every ring. The best
   one is the answer, and its confidence is the all-ring confidence.

The grid is simulated in chunks of 2,000 orientations, and each chunk is tested against all voxels of
the batch. Memory stays small and the simulation cost is shared across the batch, so give it many
voxels per call (the script uses 30).

### Why not a coarse grid with few reflections?

The adaptive search of Li and Suter (J. Appl. Cryst., 2013) scores a coarse grid using only the
lowest-order reflections, then refines. We measured whether that works on Ti-7Al, with one detector
distance and binarized images. It does not: using fewer rings does not make the confidence peak
broader, it only makes the score noisier. The share of cases that score above random orientations:

| Rings used (peaks) | 0.5 degrees off | 1 degree | 2 degrees |
|---|---|---|---|
| 3 (40) | 45% | 3% | 2% |
| 6 (100) | 89% | 31% | 8% |
| 12 (232) | 98% | 75% | 27% |
| 16 (360) | 98% | 75% | 33% |

So the first stage needs a fine grid and most of the rings. Clustering and refining only the best
groups is where the time is saved. (A 1 degree grid is 1.7 million orientations for hexagonal.)

## How the script uses it

The script keeps the structure of the original, with the same thresholds from the `missing_grains`
section of the config:

1. Load the starting reconstruction (`reconstructed_data_path`, the `.npz` from `reconstruct_nf.py`)
   and pick the voxels below `confidence_threshold` (thinned by `low_confidence_sparsing`, with the free
   surface removed if `errode_free_surface` is 1).
2. Pick a random batch of those voxels and search each one.
3. Go through the results, best first. A voxel that an earlier result in the batch already explained is
   skipped. A result below 0.75 x the threshold is a miss.
4. For a good result, test that orientation at all the remaining voxels and drop the ones it explains
   (confidence above the threshold). Keep it as a grain if it explains more than 5 voxels.
5. Stop when `coord_cutoff_scale` of the voxels are left, or after `iter_cutoff` misses in a row.
6. Search the voxels that remain. Drop results that match a grain already in the list, merge the
   rest (0.5 degrees, symmetry aware), and add them.
7. Write all grains to `<analysis_name>_grains.out`. If `re_run_and_save` is 1, run the reconstruction
   again with them.

```
python find_missing_grains.py config.yml                  # the search above
python find_missing_grains.py config.yml --legacy         # the original search
```

Other flags: `--batch-size` (30), `--grid-spacing` (1.75 degrees), `--seed`. The legacy search uses
`ori_grid_spacing` from the config instead.

From Python:

```python
em, conf, stats = nfutil.search_orientations(experiment, controller, image_stack, coordinates)
found = nfutil.find_missing_orientations(experiment, controller, image_stack,
                                         orientation_data, test_coordinates, mask)
orientation_data = nfutil.add_orientations(experiment, orientation_data, found['exp_maps'])
```

## What to expect

Tested on a Ti-7Al layer (619 FF grains, 9 CPUs):

- **Known orientations.** At voxels where the FF orientation is known, the search recovered it to
  within 0.5 degrees at 96% of 100 fresh high-confidence voxels. At weaker voxels (reference
  confidence 0.6 to 0.8) it was 62%, though at 77% it found an orientation about as good as the FF one.
- **Grains removed from the list.** With 6 grains removed, the script recovered 5 within 0.15 degrees.
  Voxels below 0.5 confidence fell from 2,504 to about 590 (701 with the full list).
- **Speed.** About 1 second per voxel, plus about 13 seconds per batch.
- **The original search.** It refines only the single best grid point and simulates the whole grid up
  front. On the same 1.75 degree grid it recovered 72% against 93%. At its default 1 degree grid it
  recovers about as many, at roughly 25 to 30 seconds per voxel (an estimate) and about 74 GB of memory.

## Cautions

- This was validated on one Ti-7Al layer. Other materials and symmetries may need a different
  `--grid-spacing`. Check recovery on voxels with a known FF orientation before relying on it.
- Choose `confidence_threshold` from your own confidence map. On Ti-7Al the mean is about 0.7, so a
  threshold of 0.7 marks nearly half the layer as low confidence.
- It finds orientations only. It does not fit position or strain, and a poor result at a voxel can mean
  a noisy voxel, not a missing grain.
- With `re_run_and_save: 1`, the final reconstruction is written under the same name as the starting
  reconstruction if the analysis name and output folder are the same, so it overwrites it. Keep a copy.
  `<analysis_name>_grains.out` is also overwritten on every run.
- The orientations written are the best found, so check them before using them as FF grains. One
  removed grain out of six was not recovered, and the legacy search also produced a grain nothing
  matched.
