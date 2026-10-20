"""Spatially adaptive (marching) NF reconstruction.

Neighboring voxels in a grain share an orientation, so most voxels do not need
to be tested against every candidate orientation. A voxel is instead tested
against only the orientations already found at its neighbors, and falls back
to the full search where that fails (grain and sample boundaries). This is the
spatially adaptive extension of Li & Suter, J. Appl. Cryst. (2013), with the
candidate orientations taken from the FF grain list.

contributing authors: lim37
"""

import itertools
import numpy as np

from .processor_functions import (
    make_chunks, multiprocessing_pool, set_multiprocessing_method,
    _test_orientation_bags_at_coordinates,
    _test_orientation_bags_at_coordinates_distributor)


def neighbor_table(mask, connectivity=6):
    """Return an (n, K) array of the voxel numbers next to each masked voxel.

    Voxels are numbered in C order over the mask (the same order as
    ``mask[mask]``). Neighbors outside the mask or the array are -1.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 3:
        raise ValueError(f'mask must be 3D, got {mask.ndim}D')
    if connectivity == 6:
        offsets = [o for a in range(3) for o in
                   (tuple(-1 if j == a else 0 for j in range(3)),
                    tuple(1 if j == a else 0 for j in range(3)))]
    elif connectivity == 26:
        offsets = [o for o in itertools.product((-1, 0, 1), repeat=3)
                   if any(o)]
    else:
        raise ValueError(f'connectivity must be 6 or 26, got {connectivity}')

    n = int(mask.sum())
    voxel_number = -np.ones(mask.shape, int)
    voxel_number[mask] = np.arange(n)
    padded = np.pad(voxel_number, 1, constant_values=-1)
    nbrs = np.empty((n, len(offsets)), int)
    for k, off in enumerate(offsets):
        window = tuple(slice(1 + o, 1 + o + size)
                       for o, size in zip(off, mask.shape))
        nbrs[:, k] = padded[window][mask]
    return nbrs


def march(mask, n_oris, test_fn, converged_confidence=0.7, drop_off=0.8,
          seed_spacing=8, connectivity=6):
    """Assign an orientation to every masked voxel by marching out from seeds.

    test_fn(voxel_numbers, bags) must return ``(best_idx, best_confidence,
    n_tests)``, testing only the orientations in ``bags[i]`` at voxel
    ``voxel_numbers[i]``.

    1. Seeds on a sparse regular grid are tested against all orientations.
    2. A voxel next to voxels whose confidence reached
       ``converged_confidence`` is tested against only their orientations. It
       is accepted if its confidence is at least ``drop_off`` times the best
       neighbor's, otherwise it is put aside.
    3. When no voxel can be reached, the voxels put aside (and any cut off
       from the seeds) are tested against all orientations, and the march
       continues from the new converged voxels.

    Returns ``idx, confidence, stats`` where idx and confidence are over the
    masked voxels, in the same order as ``mask[mask]``.
    """
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    nbrs = neighbor_table(mask, connectivity)
    valid_nbr = nbrs >= 0
    everything = np.arange(n_oris)

    idx = np.full(n, -1, int)
    confidence = np.zeros(n)
    assigned = np.zeros(n, bool)
    spreading = np.zeros(n, bool)   # converged, so neighbors may copy it
    put_aside = np.zeros(n, bool)   # failed the neighbor test
    stats = dict(n_voxels=n, n_tests=0, n_seed=0, n_global=0, n_local=0,
                 n_local_failed=0, n_iterations=0)

    def run_global(voxels):
        best, conf, n_tests = test_fn(voxels, [everything] * len(voxels))
        idx[voxels] = best
        confidence[voxels] = conf
        assigned[voxels] = True
        spreading[voxels] = conf >= converged_confidence
        stats['n_tests'] += n_tests

    # Seeds on a sparse regular grid
    index = np.indices(mask.shape)[:, mask]
    seeds = np.where(np.all(index % seed_spacing == 0, axis=0))[0]
    if len(seeds) == 0:
        centre = np.array(mask.shape)[:, None]/2.
        seeds = np.array([np.argmin(((index - centre)**2).sum(0))])
    run_global(seeds)
    stats['n_seed'] = len(seeds)

    while not assigned.all():
        stats['n_iterations'] += 1
        # Voxels next to a converged voxel that have not been tried yet
        near = (spreading[nbrs] & valid_nbr).any(axis=1)
        frontier = np.where(near & ~assigned & ~put_aside)[0]
        if len(frontier) > 0:
            nb = nbrs[frontier]
            ok = spreading[nb] & (nb >= 0)
            bags = [np.unique(idx[row[o]]) for row, o in zip(nb, ok)]
            reference = np.where(ok, confidence[nb], 0.).max(axis=1)
            best, conf, n_tests = test_fn(frontier, bags)
            stats['n_tests'] += n_tests
            accept = conf >= drop_off*reference
            take = frontier[accept]
            idx[take] = best[accept]
            confidence[take] = conf[accept]
            assigned[take] = True
            spreading[take] = conf[accept] >= converged_confidence
            put_aside[frontier[~accept]] = True
            stats['n_local'] += int(accept.sum())
            stats['n_local_failed'] += int((~accept).sum())
        else:
            leftover = np.where(~assigned)[0]
            run_global(leftover)
            stats['n_global'] += len(leftover)

    return idx, confidence, stats


def reconstruct_spatially_adaptive(experiment, controller, image_stack,
                                   orientation_data, test_coordinates, mask,
                                   converged_confidence=0.7, drop_off=0.8,
                                   seed_spacing=8, connectivity=6,
                                   parallel_min_voxels=2000):
    """Drop-in alternative to ``test_orientations_at_coordinates``.

    ``orientation_data`` is the output of ``precompute_diffraction_data`` and
    ``test_coordinates``/``mask`` are those returned by
    ``Experiment.generate_test_coordinates``. Returns ``raw_exp_maps,
    raw_confidence, raw_idx`` like ``test_orientations_at_coordinates`` (no
    refinement), plus a stats dictionary.

    Batches smaller than ``parallel_min_voxels`` are tested in this process,
    since starting a pool costs more than they do.
    """
    ncpus = controller.get_process_count()
    n_oris = np.shape(orientation_data[0])[0]

    def test_fn(voxels, bags):
        coords = test_coordinates[voxels]
        n = len(voxels)
        if ncpus == 1 or n < parallel_min_voxels:
            best, conf, n_tests, _, _ = _test_orientation_bags_at_coordinates(
                experiment, image_stack, orientation_data, coords, bags,
                start=0, stop=n)
            return best, conf, n_tests
        chunks, starts, stops, _ = make_chunks(
            n, ncpus, controller.get_chunk_size())
        best = np.zeros(n, int)
        conf = np.zeros(n)
        n_tests = 0
        state = (starts, stops, experiment, image_stack, orientation_data,
                 coords, bags)
        set_multiprocessing_method(controller.multiprocessing_start_method)
        with multiprocessing_pool(ncpus, state) as pool:
            for b, c, t, start, stop in pool.imap_unordered(
                    _test_orientation_bags_at_coordinates_distributor, chunks):
                best[start:stop] = b
                conf[start:stop] = c
                n_tests += t
            pool.close()
            pool.join()
        return best, conf, n_tests

    idx, confidence, stats = march(
        mask, n_oris, test_fn, converged_confidence, drop_off, seed_spacing,
        connectivity)
    stats['n_exhaustive_tests'] = stats['n_voxels']*n_oris
    raw_exp_maps = np.asarray(orientation_data[0]).reshape(n_oris, 3)[idx]
    return raw_exp_maps, confidence, idx, stats
