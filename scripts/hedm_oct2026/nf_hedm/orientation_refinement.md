# Orientation refinement in NF-HEDM

An FF grain's orientation is only an estimate of the orientation in the NF volume. Refinement moves
the orientation at each voxel to the one that fits the NF images best, which raises the confidence
and gives a map of how much orientations vary inside grains. `reconstruct_and_refine_nf.py` does it
for a whole layer, and the orientation search and `find_missing_grains.py` use the same code.

## How a voxel is refined

The **confidence** of an orientation at a voxel is the fraction of its simulated Bragg peaks that land
on lit pixels (see `find_missing_grains.md`). Refinement is a pattern search for the orientation that
maximizes it:

1. At a search radius, score a 3x3x3 grid of small rotations (minus the radius, zero, plus the radius
   about each axis) around the current best orientation.
2. Move to the best of the 27 and score a new grid around it, while that improves the confidence.
3. Repeat with a smaller radius. The default radii run from 5 degrees down to 0.1 degrees.

Starting from an FF grain, an orientation moves by about 0.1 degrees (at most about 0.2) and the
confidence rises by about 0.09.

## What is faster, with the same results

Refining one voxel takes about 0.04 s on one CPU, down from 0.24 s, and gives exactly the same
orientations and confidences (compared bit for bit on 60 voxels).

- 92% of the old time was simulating the peaks of each trial orientation. The simulation rebuilt the
  list of symmetry-equivalent reflections on every call and computed pixel sizes that nothing used.
  `nfutil/simulation.py` returns exactly the angles `simulateGVecs` does without that. This is 3 to 4
  times faster per orientation, checked identical on 300 random orientations for Ti-7Al (hexagonal)
  and lowO (cubic).
- The pattern search no longer scores a point twice. The center of every grid is the current best,
  whose score is known, and when a pass does not move there is no second pass over the same grid.
  That is 314 orientations per voxel instead of 595.

`simulateGVecs` has a quirk that is reproduced exactly: when some reflections have no solution (NaN
angles), it keeps the NaN rows and drops the last real peaks. On Ti-7Al this affects 45% of the
orientations, about 4 of 362 peaks each, and changes a confidence by at most 0.008. It is left as it
is so that no score changes.

## Refining along the marching

`reconstruct_nf.py --spatially-adaptive` already tests each voxel against its neighbors' grains first.
Refinement can do the same. Neighboring voxels in a grain have nearly the same orientation, so a voxel
can start from its neighbors' refined orientations and needs only a short search. Because each voxel
adjusts its neighbor's refined orientation, the orientation can drift across a grain, following a
gradient.

1. Seeds on a sparse grid (every 8th voxel) are tested against every FF grain, and the best is refined
   with radii 1, 0.5, 0.25, 0.1 degrees.
2. A voxel next to voxels whose confidence reached 0.7 scores their refined orientations. If the best is
   at least 0.8 times the best neighbor's confidence, it is refined with radii 0.25 and 0.1 degrees.
3. Voxels that fail, or are cut off from the seeds, are treated like seeds, and the march continues.

```
python reconstruct_and_refine_nf.py config.yml --spatially-adaptive
```

The options `--converged-confidence` (0.7), `--drop-off` (0.8) and `--seed-spacing` (8) are as in
`reconstruct_nf.py`. From Python it is `nfutil.reconstruct_and_refine_spatially_adaptive`, which returns
the same arrays as `test_orientations_at_coordinates(..., refine_yes_no=1, return_misorientation=1)`
plus a stats dictionary.

## What to expect

Ti-7Al, 619 FF grains for the initial scan and 546 for the final scan, 9 CPUs, one layer. "Alone" means
refining every voxel on its own from the best FF grain.

| Layer | Time alone, along the marching | Mean confidence before refining | Alone, along the marching |
|---|---|---|---|
| Initial scan, tomo mask (39,762 voxels) | 272 s, 50 s (5.5x) | 0.710 | 0.797, 0.797 |
| Final scan, initial scan's mask (39,762) | 312 s, 57 s (5.4x) | 0.703 | 0.787, 0.786 |
| Initial scan, no mask (15,625) | 111 s, 39 s (2.9x) | 0.595 | 0.665, 0.663 |

- The orientations agree to 0.05 degrees at the 90th percentile, and both land the same distance from the
  FF orientation (median 0.10 degrees).
- About 7% of voxels end on a different grain. These are weakly determined voxels (mean confidence about
  0.74 against 0.80 elsewhere), where another grain fits about as well. Neither method is better on
  average: the confidence is higher with one at about as many voxels as with the other.
- On the unmasked layer, where more voxels are weak, the default loses a little (0.0023). A converged
  confidence of 0.8 removes that and is still 2.2x faster:

| Converged confidence, drop-off | Time (111 s alone) | Mean confidence (0.6648 alone) | Voxels more than 0.05 worse |
|---|---|---|---|
| 0.6, 0.8 | 40 s | 0.6570 | 6.1% |
| 0.7, 0.8 (default) | 39 s | 0.6625 | 2.9% |
| 0.7, 0.9 | 41 s | 0.6638 | 2.0% |
| 0.8, 0.8 | 50 s | 0.6655 | 0.3% |

## Cautions

- Checked on Ti-7Al, one layer at a time. The local search uses radii of 0.25 and 0.1 degrees and
  assumes neighbors are already close, which holds here (refined orientations are 0.1 degrees from the
  FF grain). Orientations that change faster than that from voxel to voxel need larger `local_radii`.
- A refined orientation is the best fit to the images, not a unique answer. At weak voxels a different
  orientation can fit equally well.
- `test_orientations_at_coordinates` with refinement splits the work over voxels when there are more
  voxels than grains, and over grains (refining the best of each) when there are fewer. They are
  different methods and can pick different grains. Compare like with like.
