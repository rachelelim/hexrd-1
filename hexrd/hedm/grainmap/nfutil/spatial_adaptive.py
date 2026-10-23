"""Spatially adaptive (marching) NF reconstruction.

Neighboring voxels in a grain share an orientation, so most voxels do not need
to be tested against every candidate grain. A voxel is instead tested against
only the grains already found at its neighbors, and falls back to testing every
FF grain where that fails (grain and sample boundaries). This is the spatially
adaptive extension of Li & Suter, J. Appl. Cryst. (2013), with the candidate
orientations taken from the FF grain list. Nothing outside that list can be
found: the fallback tests the listed grains, it does not search orientation
space.

contributing authors: lim37
"""

import itertools
import numpy as np

from .orientation_search import misorientation_deg
from .processor_functions import (
    make_chunks, multiprocessing_pool, run_chunked, set_multiprocessing_method,
    _refine_from_candidates, _refine_from_candidates_distributor,
    _refine_orientations_at_coordinates, _refine_orientations_distributor,
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

    1. Seeds on a sparse regular grid are tested against every candidate.
    2. A voxel next to voxels whose confidence reached
       ``converged_confidence`` is tested against only their orientations. It
       is accepted if its confidence is at least ``drop_off`` times the best
       neighbor's, otherwise it is put aside.
    3. When no voxel can be reached, the voxels put aside (and any cut off
       from the seeds) are tested against every candidate, and the march
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


def _bag_test_function(experiment, controller, image_stack, orientation_data,
                       test_coordinates, parallel_min_voxels):
    """test_fn for ``march``: test a bag of candidates at each voxel, in the pool
    for large batches and in this process for small ones."""
    ncpus = controller.get_process_count()

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

    return test_fn


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
    n_oris = np.shape(orientation_data[0])[0]
    test_fn = _bag_test_function(
        experiment, controller, image_stack, orientation_data, test_coordinates,
        parallel_min_voxels)

    idx, confidence, stats = march(
        mask, n_oris, test_fn, converged_confidence, drop_off, seed_spacing,
        connectivity)
    stats['n_all_grains_tests'] = stats['n_voxels']*n_oris
    raw_exp_maps = np.asarray(orientation_data[0]).reshape(n_oris, 3)[idx]
    return raw_exp_maps, confidence, idx, stats


def march_refined(mask, global_fn, local_fn, converged_confidence=0.7,
                  drop_off=0.8, seed_spacing=8, connectivity=6):
    """Refine an orientation at every masked voxel by marching out from seeds.

    The same march as ``march``, except that a voxel carries a refined
    orientation, not an index into a list, so orientations follow gradients
    inside a grain.

    global_fn(voxel_numbers) must return ``(exp_maps, confidence, origin,
    n_tests, n_simulated)``: the best of the candidate grains at each voxel,
    refined. ``origin`` is the grain it came from.

    local_fn(voxel_numbers, candidates, reference) must return ``(exp_maps,
    confidence, pick, accepted, n_simulated)``. ``candidates[i]`` holds the
    refined orientations of the converged neighbors of voxel i and
    ``reference[i]`` the best of their confidences. The best candidate is
    refined if its confidence is at least ``drop_off`` times the reference, and
    ``accepted`` says whether it was. ``pick`` is the candidate used.

    1. Seeds on a sparse regular grid go through global_fn.
    2. A voxel next to voxels whose confidence reached ``converged_confidence``
       goes through local_fn with their orientations. Voxels that are not
       accepted are put aside.
    3. When no voxel can be reached, the voxels put aside (and any cut off from
       the seeds) go through global_fn, and the march continues.

    Returns ``exp_maps, confidence, origin, stats``, over the masked voxels in
    the same order as ``mask[mask]``.
    """
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    nbrs = neighbor_table(mask, connectivity)
    valid_nbr = nbrs >= 0

    exp_maps = np.zeros((n, 3))
    confidence = np.zeros(n)
    origin = np.full(n, -1, int)
    assigned = np.zeros(n, bool)
    spreading = np.zeros(n, bool)   # converged, so neighbors may start from it
    put_aside = np.zeros(n, bool)   # failed the neighbor test
    stats = dict(n_voxels=n, n_grain_tests=0, n_simulated=0, n_seed=0,
                 n_global=0, n_local=0, n_local_failed=0, n_iterations=0)

    def run_global(voxels):
        em, conf, org, n_tests, n_simulated = global_fn(voxels)
        exp_maps[voxels] = em
        confidence[voxels] = conf
        origin[voxels] = org
        assigned[voxels] = True
        spreading[voxels] = conf >= converged_confidence
        stats['n_grain_tests'] += n_tests
        stats['n_simulated'] += n_simulated

    index = np.indices(mask.shape)[:, mask]
    seeds = np.where(np.all(index % seed_spacing == 0, axis=0))[0]
    if len(seeds) == 0:
        centre = np.array(mask.shape)[:, None]/2.
        seeds = np.array([np.argmin(((index - centre)**2).sum(0))])
    run_global(seeds)
    stats['n_seed'] = len(seeds)

    while not assigned.all():
        stats['n_iterations'] += 1
        near = (spreading[nbrs] & valid_nbr).any(axis=1)
        frontier = np.where(near & ~assigned & ~put_aside)[0]
        if len(frontier) > 0:
            nb = nbrs[frontier]
            ok = spreading[nb] & (nb >= 0)
            candidates = [exp_maps[row[o]] for row, o in zip(nb, ok)]
            candidate_origin = [origin[row[o]] for row, o in zip(nb, ok)]
            reference = np.where(ok, confidence[nb], 0.).max(axis=1)
            em, conf, pick, accepted, n_simulated = local_fn(frontier, candidates, reference)
            take = frontier[accepted]
            exp_maps[take] = em[accepted]
            confidence[take] = conf[accepted]
            origin[take] = [candidate_origin[i][pick[i]] for i in np.where(accepted)[0]]
            assigned[take] = True
            spreading[take] = conf[accepted] >= converged_confidence
            put_aside[frontier[~accepted]] = True
            stats['n_simulated'] += n_simulated
            stats['n_local'] += int(accepted.sum())
            stats['n_local_failed'] += int((~accepted).sum())
        else:
            leftover = np.where(~assigned)[0]
            run_global(leftover)
            stats['n_global'] += len(leftover)

    return exp_maps, confidence, origin, stats


def boundary_band(origin, nbrs, width=2):
    """Voxels at or near a boundary between grains, and the grains seen near each.

    A boundary voxel has a neighbor that descends from a different grain.
    ``width`` 1 is the boundary voxels, 2 adds the voxels touching them, and so
    on. Returns ``voxels`` and ``bags``, where ``bags[i]`` holds the grains
    (``origin`` values) found within ``width`` steps of ``voxels[i]``.
    """
    origin = np.asarray(origin)
    valid = nbrs >= 0
    differs = (valid & (origin[np.where(valid, nbrs, 0)] != origin[:, None])).any(axis=1)
    band = differs.copy()
    for _ in range(width - 1):
        band = band | (band[np.where(valid, nbrs, 0)] & valid).any(axis=1)
    seen = [{int(g)} for g in origin]
    for _ in range(width):
        seen = [set().union(seen[i], *(seen[j] for j in row[row >= 0])) for i, row in enumerate(nbrs)]
    voxels = np.where(band)[0]
    return voxels, [np.array(sorted(seen[i])) for i in voxels]


def refine_boundaries(exp_maps, confidence, origin, nbrs, band_fn, width=2):
    """Retest the voxels at and near grain boundaries against the grains around them.

    The march lets a grain spread into weakly determined voxels along the grid
    axes, which leaves boundaries made of straight steps. Here every voxel in
    ``boundary_band`` is tested against the grains found within ``width`` steps
    of it, and the best is refined, as when each voxel is refined on its own.

    band_fn(voxels, bags) must return ``(exp_maps, confidence, origin, n_tests,
    n_simulated)`` like the global function of ``march_refined``. The arrays are
    changed in place. Returns a dict of statistics.
    """
    voxels, bags = boundary_band(origin, nbrs, width)
    stats = dict(n_band=len(voxels), n_band_changed=0, n_band_tests=0, n_band_simulated=0)
    if len(voxels) == 0:
        return stats
    before = origin[voxels].copy()
    em, conf, org, n_tests, n_simulated = band_fn(voxels, bags)
    exp_maps[voxels] = em
    confidence[voxels] = conf
    origin[voxels] = org
    stats.update(n_band_changed=int((org != before).sum()), n_band_tests=n_tests, n_band_simulated=n_simulated)
    return stats


def reconstruct_and_refine_spatially_adaptive(
        experiment, controller, image_stack, orientation_data,
        test_coordinates, mask, converged_confidence=0.7, drop_off=0.8,
        seed_spacing=8, connectivity=6, global_radii=(1.0, 0.5, 0.25, 0.1),
        local_radii=(0.25, 0.1), parallel_min_voxels=2000,
        parallel_min_voxels_refine=100, boundary_width=2, boundary_passes=2):
    """Drop-in alternative to ``test_orientations_at_coordinates`` with
    ``refine_yes_no=1, return_misorientation=1``.

    Seeds, and voxels the neighbors cannot explain, are tested against every
    grain and the best is refined with the pattern search over ``global_radii``
    (degrees). Every other voxel starts from the refined orientations of its
    neighbors and refines the best with ``local_radii``. A voxel that is
    refined from its neighbor's refined orientation follows orientation
    gradients inside a grain, and needs a short search.

    Returns ``raw_exp_maps, raw_confidence, raw_idx, raw_misorientation,
    stats``. raw_idx is the grain each voxel's orientation descends from and
    raw_misorientation the distance (degrees) from that grain's orientation.
    Voxels of a refinement batch smaller than ``parallel_min_voxels_refine``
    are refined in this process.

    The march lets grains spread into weak voxels along the grid axes, so the
    boundaries come out as straight steps. With ``boundary_width`` > 0 the
    voxels within that many steps of a boundary are retested against the grains
    around them and refined (see ``refine_boundaries``), ``boundary_passes``
    times. 0 turns it off.
    """
    grain_exp_maps = np.asarray(orientation_data[0], float).reshape(-1, 3)
    n_oris = len(grain_exp_maps)
    everything = np.arange(n_oris)
    test_fn = _bag_test_function(
        experiment, controller, image_stack, orientation_data, test_coordinates,
        parallel_min_voxels)

    def pool_or_inline(n, tail, distributor, inline):
        if n < parallel_min_voxels_refine or controller.get_process_count() == 1:
            return [inline(*tail, start=0, stop=n)]
        return run_chunked(controller, n, controller.get_chunk_size(), tail, distributor, inline)

    def global_fn(voxels, bags=None):
        best, conf, n_tests = test_fn(voxels, [everything]*len(voxels) if bags is None else bags)
        coords = test_coordinates[voxels]
        results = pool_or_inline(
            len(voxels), (experiment, image_stack, coords, grain_exp_maps[best], tuple(global_radii)),
            _refine_orientations_distributor, _refine_orientations_at_coordinates)
        em = np.concatenate([r[0] for r in results])
        refined = np.concatenate([r[1] for r in results])
        return em, refined, best, n_tests, sum(r[2] for r in results)

    def local_fn(voxels, candidates, reference):
        coords = test_coordinates[voxels]
        results = pool_or_inline(
            len(voxels), (experiment, image_stack, coords, candidates, reference, drop_off, tuple(local_radii)),
            _refine_from_candidates_distributor, _refine_from_candidates)
        return (np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results]),
                np.concatenate([r[2] for r in results]), np.concatenate([r[3] for r in results]),
                sum(r[4] for r in results))

    exp_maps, confidence, origin, stats = march_refined(
        mask, global_fn, local_fn, converged_confidence, drop_off, seed_spacing, connectivity)
    if boundary_width > 0:
        nbrs = neighbor_table(mask, connectivity)
        stats.update(n_band=0, n_band_changed=0, n_band_tests=0, n_band_simulated=0)
        for _ in range(boundary_passes):
            for key, value in refine_boundaries(exp_maps, confidence, origin, nbrs, global_fn, boundary_width).items():
                stats[key] += value
        stats['n_simulated'] += stats['n_band_simulated']
        stats['n_grain_tests'] += stats['n_band_tests']

    q_sym = experiment.plane_data.q_sym
    misorientation = np.zeros(len(origin))
    for g in np.unique(origin):
        members = np.where(origin == g)[0]
        misorientation[members] = misorientation_deg(grain_exp_maps[g], exp_maps[members], q_sym)
    return exp_maps, confidence, origin, misorientation, stats
