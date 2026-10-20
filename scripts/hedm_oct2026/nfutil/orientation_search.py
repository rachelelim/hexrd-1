"""Orientation-space search for NF reconstruction.

Finds the orientation at a voxel without a candidate list, so grains missing
from the FF grain list can be found. It follows the idea in Li & Suter, J.
Appl. Cryst. (2013): score a grid over the fundamental zone, keep the best
regions, and refine only those, instead of refining every grid point.

Measured on Ti-7Al, the confidence peak around a true orientation is narrow
(a 1 degree error already costs most of it) and using fewer rings does not
broaden it, so the first stage cannot use a coarse grid scored with a few
low-order reflections. It uses a ~1.75 degree grid scored with most of the rings,
clusters the best grid points, and refines the survivors with the coarse-to-fine
pattern search used elsewhere in nfutil.

contributing authors: lim37
"""

import copy
import timeit

import numpy as np

from hexrd import rotations

from .processor_functions import (
    run_chunked,
    _grid_top_candidates, _grid_top_candidates_distributor,
    _refine_orientations_at_coordinates, _refine_orientations_distributor,
    _precompute_diffraction_data_of_many_orientations)
from .utilities import uniform_fundamental_zone_sampling


def ring_subset_experiment(experiment, n_rings):
    """A copy of ``experiment`` that simulates only the first ``n_rings`` rings.

    Rings are ordered by scattering angle, so this keeps the lowest-order
    reflections. ``n_rings=None`` (or at least all of them) returns the
    experiment itself.
    """
    two_theta = experiment.plane_data.getTTh()
    if n_rings is None or n_rings >= len(two_theta):
        return experiment
    subset = copy.copy(experiment)
    subset.plane_data = copy.deepcopy(experiment.plane_data)
    subset.plane_data.tThMax = two_theta[n_rings - 1] + 1e-6
    return subset


def misorientation_deg(exp_map, exp_maps, q_sym):
    """Misorientation (degrees, symmetry aware) of one exp-map from many."""
    q_one = np.reshape(
        rotations.quatOfExpMap(np.asarray(exp_map, float).reshape(3, 1)), (4, 1))
    q_many = np.reshape(
        rotations.quatOfExpMap(np.atleast_2d(exp_maps).T), (4, -1))
    return np.degrees(np.atleast_1d(
        rotations.misorientation(q_one, q_many, (q_sym,))[0]))


def separate_candidates(exp_maps, confidence, q_sym, min_separation_deg, n_max):
    """Keep the best candidate of each cluster, best first.

    A candidate is dropped if it is within ``min_separation_deg`` of a better
    one that was kept. Returns indices into ``exp_maps``.
    """
    order = np.argsort(-confidence, kind='stable')
    kept = []
    for i in order:
        if kept and misorientation_deg(
                exp_maps[i], exp_maps[kept], q_sym).min() < min_separation_deg:
            continue
        kept.append(i)
        if len(kept) == n_max:
            break
    return np.array(kept, int)


def merge_orientations(exp_maps, confidence, q_sym, tol_deg):
    """Collapse orientations within ``tol_deg`` of each other, best first.

    Returns indices into ``exp_maps`` of the orientations kept.
    """
    if len(exp_maps) == 0:
        return np.array([], int)
    return separate_candidates(
        np.asarray(exp_maps), np.asarray(confidence), q_sym, tol_deg, len(exp_maps))


def search_orientations(experiment, controller, image_stack, coordinates,
                        grid_spacing_deg=1.75, grid_rings=12, n_candidates=200,
                        n_refine=20, n_final=4, coarse_rings=12,
                        coarse_radii=(1.5, 1.0, 0.5), fine_radii=(0.5, 0.25, 0.1),
                        grid_chunk=2000, min_separation_deg=None):
    """Find the best orientation at each coordinate by searching orientation space.

    1. Score a grid of the fundamental zone (``grid_spacing_deg``) at every
       coordinate using the first ``grid_rings`` rings, keep the best
       ``n_candidates`` grid points per coordinate.
    2. Cluster those (clusters closer than ``min_separation_deg``, default the
       grid spacing) and keep the best ``n_refine`` clusters.
    3. Refine each with the pattern search over ``coarse_radii`` (degrees),
       using ``coarse_rings`` rings, and keep the best ``n_final``.
    4. Refine those with ``fine_radii`` using every ring.

    ``coordinates`` is (V, 3). Returns ``exp_maps`` (V, 3), ``confidence`` (V,)
    and a stats dictionary. Batch many coordinates in one call: the grid
    diffraction data is simulated once per call, not once per coordinate.
    """
    coordinates = np.atleast_2d(np.asarray(coordinates, float))
    n_vox = len(coordinates)
    q_sym = experiment.plane_data.q_sym
    min_sep = grid_spacing_deg if min_separation_deg is None else min_separation_deg
    stats = dict(n_voxels=n_vox)
    t0 = timeit.default_timer()

    # 1. Score the grid
    quats = uniform_fundamental_zone_sampling(
        experiment.point_group_number, grid_spacing_deg)
    grid = np.ascontiguousarray(rotations.expMapOfQuat(quats.T).T)
    grid_experiment = ring_subset_experiment(experiment, grid_rings)
    results = run_chunked(
        controller, len(grid), grid_chunk,
        (grid_experiment, image_stack, grid, coordinates, n_candidates),
        _grid_top_candidates_distributor, _grid_top_candidates)
    all_idx = np.concatenate([r[0] for r in results], axis=1)
    all_conf = np.concatenate([r[1] for r in results], axis=1)
    stats['n_grid'] = len(grid)
    stats['n_grid_tests'] = int(sum(r[2] for r in results))
    t1 = timeit.default_timer()
    stats['grid_s'] = t1 - t0

    # 2. Cluster the best grid points of each coordinate
    task_voxel = []
    task_start = []
    for v in range(n_vox):
        k = min(n_candidates, all_conf.shape[1])
        best = np.argpartition(-all_conf[v], k - 1)[:k]
        cand = grid[all_idx[v, best]]
        keep = separate_candidates(cand, all_conf[v, best], q_sym, min_sep, n_refine)
        task_voxel.extend([v]*len(keep))
        task_start.extend(cand[keep])
    task_voxel = np.array(task_voxel)
    task_start = np.array(task_start)
    task_coords = coordinates[task_voxel]
    stats['n_refined_candidates'] = len(task_voxel)

    # 3. Coarse refinement of every cluster
    coarse_experiment = ring_subset_experiment(experiment, coarse_rings)
    results = run_chunked(
        controller, len(task_voxel), -1,
        (coarse_experiment, image_stack, task_coords, task_start, tuple(coarse_radii)),
        _refine_orientations_distributor, _refine_orientations_at_coordinates)
    em = np.concatenate([r[0] for r in results])
    conf = np.concatenate([r[1] for r in results])
    n_evals = sum(r[2] for r in results)
    survivors = []
    for v in range(n_vox):
        mine = np.where(task_voxel == v)[0]
        survivors.extend(mine[np.argsort(-conf[mine])[:n_final]])
    survivors = np.array(survivors, int)
    t2 = timeit.default_timer()
    stats['coarse_refine_s'] = t2 - t1

    # 4. Fine refinement of the survivors with every ring
    results = run_chunked(
        controller, len(survivors), -1,
        (experiment, image_stack, task_coords[survivors], em[survivors], tuple(fine_radii)),
        _refine_orientations_distributor, _refine_orientations_at_coordinates)
    em_fine = np.concatenate([r[0] for r in results])
    conf_fine = np.concatenate([r[1] for r in results])
    n_evals += sum(r[2] for r in results)
    stats['n_refine_evals'] = int(n_evals)
    stats['fine_refine_s'] = timeit.default_timer() - t2

    out_em = np.zeros((n_vox, 3))
    out_conf = np.zeros(n_vox)
    owner = task_voxel[survivors]
    for v in range(n_vox):
        mine = np.where(owner == v)[0]
        j = mine[np.argmax(conf_fine[mine])]
        out_em[v] = em_fine[j]
        out_conf[v] = conf_fine[j]
    stats['total_s'] = timeit.default_timer() - t0
    return out_em, out_conf, stats


def find_missing_orientations(experiment, controller, image_stack, orientation_data,
                              test_coordinates, mask, confidence_below=0.5,
                              max_voxels=60, min_gain=0.1, merge_tol_deg=1.0,
                              **search_kwargs):
    """Look for orientations missing from the candidate list.

    Reconstructs with the spatially adaptive search over the current
    candidates, then searches orientation space at up to ``max_voxels``
    voxels (spread over the layer) whose confidence is below
    ``confidence_below``. An orientation is kept if it beats the voxel's
    current confidence by ``min_gain``, and orientations within
    ``merge_tol_deg`` of each other are merged, keeping the best.

    Returns a dictionary with ``exp_maps`` and ``confidence`` of the new
    orientations, ``voxels`` (the masked-voxel numbers they were found at),
    ``current_confidence`` there, and the search ``stats``. Add them with
    ``add_orientations`` and reconstruct again to use them.
    """
    from .spatial_adaptive import reconstruct_spatially_adaptive

    _, confidence, _, _ = reconstruct_spatially_adaptive(
        experiment, controller, image_stack, orientation_data,
        test_coordinates, mask)
    low = np.where(confidence < confidence_below)[0]
    empty = dict(exp_maps=np.zeros((0, 3)), confidence=np.zeros(0),
                 voxels=np.zeros(0, int), current_confidence=np.zeros(0),
                 stats={}, n_low_confidence=len(low))
    if len(low) == 0:
        return empty
    if len(low) > max_voxels:
        low = low[np.linspace(0, len(low) - 1, max_voxels).astype(int)]
    em, conf, stats = search_orientations(
        experiment, controller, image_stack, test_coordinates[low], **search_kwargs)
    good = conf >= confidence[low] + min_gain
    if not good.any():
        empty['stats'] = stats
        return empty
    em, conf, voxels = em[good], conf[good], low[good]
    keep = merge_orientations(em, conf, experiment.plane_data.q_sym, merge_tol_deg)
    return dict(exp_maps=em[keep], confidence=conf[keep], voxels=voxels[keep],
                current_confidence=confidence[voxels[keep]], stats=stats,
                n_low_confidence=int((confidence < confidence_below).sum()))


def add_orientations(experiment, orientation_data, new_exp_maps):
    """Append orientations to the experiment's candidate list.

    Updates ``experiment.exp_maps``, ``remap`` and ``n_grains`` in place (new
    grain ids continue after the existing ones) and returns the precomputed
    orientation data extended with the new orientations.
    """
    new_exp_maps = np.atleast_2d(new_exp_maps)
    if len(new_exp_maps) == 0:
        return orientation_data
    exp_maps, angles, rMat_ss, gvec_cs, rMat_c = orientation_data
    new = _precompute_diffraction_data_of_many_orientations(experiment, new_exp_maps)
    experiment.exp_maps = np.vstack([experiment.exp_maps, new_exp_maps])
    experiment.remap = np.concatenate(
        [experiment.remap, experiment.remap.max() + 1 + np.arange(len(new_exp_maps))])
    experiment.n_grains = len(experiment.exp_maps)
    return (np.vstack([np.asarray(exp_maps).reshape(-1, 3), new_exp_maps]),
            list(angles) + list(new[1]), list(rMat_ss) + list(new[2]),
            list(gvec_cs) + list(new[3]), list(rMat_c) + list(new[4]))
