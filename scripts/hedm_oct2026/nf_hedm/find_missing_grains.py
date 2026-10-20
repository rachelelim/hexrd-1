#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find missing grains in a reconstructed volume using HEXRD near-field grain mapping.

Voxels with low confidence in the starting reconstruction are searched for
orientations that are missing from the FF grain list. By default each voxel is
searched with nfutil.search_orientations, which scores a fundamental-zone grid
in chunks, clusters the best grid points and refines each cluster. With
--legacy the original search is used instead: it simulates the whole
fundamental-zone grid up front (ori_grid_spacing, about 74 GB for a 1 degree
grid of Ti-7Al), tests it at one voxel at a time and refines only the single
best grid point.

Both modes share the thresholds in the missing_grains section of the config
(confidence_threshold, low_confidence_sparsing, errode_free_surface,
coord_cutoff_scale, iter_cutoff, re_run_and_save) and write the same grains.out.

Authors: dcp5303, ken38, seg246, lim37
"""

import argparse
import logging
import os
import time
from typing import Any, Tuple

import numpy as np
from hexrd import instrument, constants, rotations
from hexrd.transforms import xfcapi

from hexrd.hedm import nf_config
import sys

# nfutil lives in the parent (hedm_oct2026) folder
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nfutil
from nfutil.orientation_search import merge_orientations, misorientation_deg as orientation_difference_deg

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- Utility Functions ---


def load_configuration(config_path: str) -> Any:
    """Load and return the configuration object."""
    configuration = nf_config.open_file(config_path)[0]
    return configuration


def setup_experiment(configuration: Any) -> Tuple[Any, np.ndarray, Any]:
    """Generate experiment, image stack, and controller from configuration."""
    experiment = nfutil.Experiment(configuration)
    image_stack = experiment.image_stack
    controller = nfutil.build_controller(configuration)
    return experiment, image_stack, controller


def load_starting_reconstruction(experiment: Any) -> dict:
    """Load the starting reconstruction data."""
    return np.load(experiment.reconstructed_data_path)


def generate_orientation_grid(experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quaternion and exponential map grid for orientation sampling."""
    quats = np.transpose(
        nfutil.uniform_fundamental_zone_sampling(
            experiment.point_group_number,
            average_angular_spacing_in_deg=experiment.ori_grid_spacing
        )
    )
    exp_maps = np.array([
        2 * np.arccos(quats[0, i]) * xfcapi.unit_vector(quats[1:, i])
        for i in range(quats.shape[1])
    ])
    return quats, exp_maps


def precompute_orientations(experiment: Any, controller: Any, exp_maps: np.ndarray) -> Any:
    """Precompute diffraction data for all orientations."""
    return nfutil.precompute_diffraction_data(experiment, controller, exp_maps)


def get_low_confidence_voxels(starting_reconstruction: dict, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Get coordinates and IDs of low-confidence voxels to test."""
    return nfutil.generate_low_confidence_test_coordinates(
        starting_reconstruction,
        confidence_threshold=experiment.confidence_threshold,
        how_sparse=experiment.low_confidence_sparsing,
        errode_free_surface=experiment.errode_free_surface
    )


def merge_similar_orientations(exp_maps: np.ndarray, confidence: np.ndarray,
                               misorientation_deg: float = 0.25,
                               crystal_symmetries: np.ndarray = None) -> np.ndarray:
    """
    Merge orientations within a misorientation threshold, keeping the most
    confident of each group.

    exp_maps (n, 3) are crystal-to-sample orientations and confidence (n,) their
    confidences. crystal_symmetries (4, n_sym) are the crystal symmetry operators,
    which act on the crystal side of the orientation, so orientations that differ
    by a symmetry operation count as the same. Returns the merged exp maps (m, 3).
    """
    if len(exp_maps) == 0:
        return np.zeros((0, 3))
    if crystal_symmetries is None:
        crystal_symmetries = np.array([[1.], [0.], [0.], [0.]])
    keep = merge_orientations(
        np.asarray(exp_maps, float), np.asarray(confidence, float),
        crystal_symmetries, misorientation_deg)
    logging.info(f'Found {len(keep)} additional grains during brute force search.')
    return np.asarray(exp_maps, float)[keep]


def drop_known_orientations(exp_maps: np.ndarray, known_exp_maps: np.ndarray,
                            tolerance_deg: float = 0.5,
                            crystal_symmetries: np.ndarray = None) -> np.ndarray:
    """
    Boolean mask of the exp_maps (n, 3) that are not within tolerance_deg of any
    of known_exp_maps (m, 3), counting symmetry-equivalent orientations as the same.
    """
    if crystal_symmetries is None:
        crystal_symmetries = np.array([[1.], [0.], [0.], [0.]])
    exp_maps = np.atleast_2d(exp_maps)
    if len(known_exp_maps) == 0:
        return np.ones(len(exp_maps), bool)
    return np.array([orientation_difference_deg(e, known_exp_maps, crystal_symmetries).min() >= tolerance_deg
                     for e in exp_maps], bool)


def save_grains_out(experiment: Any, exp_maps: np.ndarray) -> None:
    """Save grains.out file with all found grains."""
    output_path = os.path.join(
        experiment.output_directory, experiment.analysis_name + '_grains.out')
    gw = instrument.GrainDataWriter(output_path)
    for gid, ori in enumerate(exp_maps):
        grain_params = np.hstack(
            [ori, constants.zeros_3, constants.identity_6x1])
        gw.dump_grain(gid, 1., 0., grain_params)
    gw.close()
    logging.info(f'Saved grains.out to {output_path}')


def rerun_and_save_reconstruction(experiment: Any, controller: Any, image_stack: np.ndarray, all_exp_maps: np.ndarray) -> None:
    """Re-run reconstruction with the original plus new grains and save outputs."""
    # Add the new grains to the experiment, giving them ids after the existing ones
    n_new = all_exp_maps.shape[0] - experiment.exp_maps.shape[0]
    new_ids = experiment.remap.max() + 1 + np.arange(n_new)
    experiment.remap = np.concatenate([experiment.remap, new_ids])
    experiment.exp_maps = all_exp_maps
    experiment.n_grains = all_exp_maps.shape[0]

    Xs, Ys, Zs, mask, test_coordinates = experiment.generate_test_coordinates()
    precomputed_orientation_data = nfutil.precompute_diffraction_data(
        experiment, controller, experiment.exp_maps
    )
    raw_exp_maps, raw_confidence, raw_idx = nfutil.test_orientations_at_coordinates(
        experiment, controller, image_stack, precomputed_orientation_data, test_coordinates
    )
    grain_map, confidence_map = nfutil.process_raw_data(
        raw_confidence, raw_idx, Xs.shape, mask=mask, id_remap=experiment.remap
    )
    nfutil.save_nf_data(
        experiment.output_directory, experiment.analysis_name,
        grain_map, confidence_map, Xs, Ys, Zs, experiment.exp_maps,
        tomo_mask=mask, id_remap=experiment.remap, save_type=['npz']
    )
    nfutil.save_nf_data_for_paraview(
        experiment.output_directory, experiment.analysis_name,
        grain_map, confidence_map, Xs, Ys, Zs, experiment.exp_maps,
        experiment.mat[experiment.material_name], tomo_mask=mask,
        id_remap=experiment.remap
    )
    logging.info('Reconstruction re-run and outputs saved.')

# --- Main Grain Search Workflow ---


def find_missing_grains_legacy(config_path: str, seed: int = None) -> None:
    """Original workflow: grid search at one voxel at a time (opt-in with --legacy)."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    configuration = load_configuration(config_path)
    experiment, image_stack, controller = setup_experiment(configuration)
    starting_reconstruction = load_starting_reconstruction(experiment)

    quats, exp_maps_grid = generate_orientation_grid(experiment)
    orientation_data_grid = precompute_orientations(
        experiment, controller, exp_maps_grid)

    original_confidence = starting_reconstruction['confidence_map']
    original_exp_maps = starting_reconstruction['ori_list']
    new_exp_maps = np.copy(original_exp_maps)

    test_coordinates, ids = get_low_confidence_voxels(
        starting_reconstruction, experiment)
    n_grains_found = 0
    n_grains_saved = 0
    no_grain_count = 0
    new_line = '\n'

    coord_cutoff = test_coordinates.shape[0] * experiment.coord_cutoff_scale

    logging.info(
        f"Testing {test_coordinates.shape[0]} coordinates against {exp_maps_grid.shape[0]} orientations.")

    # Random search for missing grains
    while test_coordinates.shape[0] > coord_cutoff:
        idx = rng.choice(test_coordinates.shape[0])
        coordinate_to_test = test_coordinates[idx, :]
        id_to_test = ids[idx]

        refined_exp_map, refined_conf, refined_idx = nfutil.test_orientations_at_coordinates(
            experiment, controller, image_stack, orientation_data_grid, coordinate_to_test, refine_yes_no=1
        )
        conf_value = refined_conf[0] if isinstance(
            refined_conf, np.ndarray) else refined_conf
        logging.info(
            f"Orientation determined at voxel (ID {id_to_test}) with {conf_value*100:.1f}% confidence.")

        if conf_value < experiment.confidence_threshold * 0.75:
            logging.info('No grain found at this voxel, moving to the next.')
            mask_idx = ids != id_to_test
            test_coordinates = test_coordinates[mask_idx]
            ids = ids[mask_idx]
            no_grain_count += 1
            logging.info(
                f'No grain found for {no_grain_count} iterations; breaking at {experiment.iter_cutoff}. {new_line}')
        else:
            n_grains_found += 1
            logging.info(f'Found a grain! Total found: {n_grains_found}')

            single_orientation_data = nfutil.precompute_diffraction_data(
                experiment, controller, refined_exp_map
            )
            exp_maps, confidence, idxs = nfutil.test_orientations_at_coordinates(
                experiment, controller, image_stack, single_orientation_data, test_coordinates
            )
            found_voxels = np.sum(confidence > experiment.confidence_threshold)
            logging.info(f'Orientation found at {found_voxels} voxels.')

            # Voxels the orientation explains leave the list, and so does the voxel
            # it was found at, even if it does not clear the threshold. Otherwise
            # the loop can pick the same voxel again and never make progress.
            mask_idx = np.logical_and(
                confidence < experiment.confidence_threshold, ids != id_to_test)
            test_coordinates = test_coordinates[mask_idx]
            ids = ids[mask_idx]

            if found_voxels > 5:
                n_grains_saved += 1
                new_exp_maps = np.vstack([new_exp_maps, refined_exp_map])
                save_grains_out(experiment, new_exp_maps)
                logging.info(f'Saved {n_grains_saved} new grains.{new_line}')
            else:
                logging.info(
                    f'Not enough voxels to consider a new grain. {new_line}')

            no_grain_count = 0

        logging.info(
            f"{test_coordinates.shape[0]} low-confidence coordinates left to test.")
        elapsed_min = (time.time() - start_time) / 60.
        logging.info(f"Elapsed time: {elapsed_min:.2f} minutes.")

        if no_grain_count == experiment.iter_cutoff:
            break

    # Brute force remaining voxels
    logging.info(
        f"Brute force: {test_coordinates.shape[0]} coordinates, {exp_maps_grid.shape[0]} orientations.")
    refined_exp_maps, refined_confidence, refined_idx = nfutil.test_orientations_at_coordinates(
        experiment, controller, image_stack, orientation_data_grid, test_coordinates,
        refine_yes_no=1
    )
    logging.info(
        f"Brute force search done. Refining orientations above {experiment.confidence_threshold} confidence.")

    # Merge similar orientations. These are the refined orientations: the grid
    # orientations they started from can be a degree away, which costs most of
    # the confidence.
    valid = refined_confidence > experiment.confidence_threshold
    final_exp_maps = merge_similar_orientations(
        np.atleast_2d(refined_exp_maps)[valid], np.atleast_1d(refined_confidence)[valid],
        misorientation_deg=0.5, crystal_symmetries=experiment.plane_data.q_sym)
    all_exp_maps = np.vstack([new_exp_maps, final_exp_maps])

    save_grains_out(experiment, all_exp_maps)

    # Optionally rerun and save reconstruction
    if experiment.re_run_and_save == 1:
        rerun_and_save_reconstruction(
            experiment, controller, image_stack, all_exp_maps)

    elapsed_min = (time.time() - start_time) / 60.
    logging.info(
        f"Total grains found: {all_exp_maps.shape[0]}. Total time: {elapsed_min:.2f} minutes.")
    logging.info("Done.")


def find_missing_grains_search(config_path: str, seed: int = None, batch_size: int = 30,
                               grid_spacing: float = 1.75, tail_batch_size: int = 200) -> None:
    """
    Workflow using nfutil.search_orientations.

    Same structure as the original: pick random low-confidence voxels, find an
    orientation at each, test a good one at all the remaining voxels and drop the
    ones it explains. The voxels are searched batch_size at a time, since the
    grid is simulated once per batch and not once per voxel, and each batch is
    then processed best result first, skipping voxels an earlier result in the
    batch already explained. The voxels left at the end are searched too, and the
    confident results are merged.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    configuration = load_configuration(config_path)
    experiment, image_stack, controller = setup_experiment(configuration)
    starting_reconstruction = load_starting_reconstruction(experiment)
    new_exp_maps = np.copy(starting_reconstruction['ori_list'])
    threshold = experiment.confidence_threshold
    search_options = dict(grid_spacing_deg=grid_spacing)

    test_coordinates, ids = get_low_confidence_voxels(
        starting_reconstruction, experiment)
    coord_cutoff = test_coordinates.shape[0] * experiment.coord_cutoff_scale
    n_grains_found = 0
    n_grains_saved = 0
    no_grain_count = 0
    new_line = '\n'
    logging.info(
        f"Searching {test_coordinates.shape[0]} low-confidence coordinates, {batch_size} at a time, "
        f"on a {grid_spacing} degree orientation grid.")

    def drop_voxel(voxel_id):
        keep = ids != voxel_id
        return test_coordinates[keep], ids[keep]

    while test_coordinates.shape[0] > coord_cutoff and no_grain_count < experiment.iter_cutoff:
        n_batch = min(batch_size, test_coordinates.shape[0])
        picked_ids = ids[rng.choice(test_coordinates.shape[0], n_batch, replace=False)]
        picked = np.array([np.where(ids == i)[0][0] for i in picked_ids])
        found_exp_maps, found_conf, stats = nfutil.search_orientations(
            experiment, controller, image_stack, test_coordinates[picked], **search_options)
        logging.info(f"Searched {n_batch} voxels in {stats['total_s']:.0f} s.")

        for k in np.argsort(-found_conf):
            if test_coordinates.shape[0] <= coord_cutoff or no_grain_count >= experiment.iter_cutoff:
                break
            id_to_test = picked_ids[k]
            if not np.any(ids == id_to_test):
                continue    # an earlier result in this batch explained it
            conf_value = found_conf[k]
            logging.info(
                f"Orientation determined at voxel (ID {id_to_test}) with {conf_value*100:.1f}% confidence.")
            if conf_value < threshold * 0.75:
                logging.info('No grain found at this voxel, moving to the next.')
                test_coordinates, ids = drop_voxel(id_to_test)
                no_grain_count += 1
                logging.info(
                    f'No grain found for {no_grain_count} iterations; breaking at {experiment.iter_cutoff}. {new_line}')
                continue

            n_grains_found += 1
            logging.info(f'Found a grain! Total found: {n_grains_found}')
            single_orientation_data = nfutil.precompute_diffraction_data(
                experiment, controller, found_exp_maps[k])
            _, confidence, _ = nfutil.test_orientations_at_coordinates(
                experiment, controller, image_stack, single_orientation_data, test_coordinates)
            found_voxels = np.sum(confidence > threshold)
            logging.info(f'Orientation found at {found_voxels} voxels.')
            keep = np.logical_and(confidence < threshold, ids != id_to_test)
            test_coordinates = test_coordinates[keep]
            ids = ids[keep]
            if found_voxels > 5:
                n_grains_saved += 1
                new_exp_maps = np.vstack([new_exp_maps, found_exp_maps[k]])
                save_grains_out(experiment, new_exp_maps)
                logging.info(f'Saved {n_grains_saved} new grains.{new_line}')
            else:
                logging.info(
                    f'Not enough voxels to consider a new grain. {new_line}')
            no_grain_count = 0

        logging.info(
            f"{test_coordinates.shape[0]} low-confidence coordinates left to test.")
        logging.info(f"Elapsed time: {(time.time() - start_time) / 60.:.2f} minutes.")

    # Search the voxels that are left, and merge what is confident
    final_exp_maps = np.zeros((0, 3))
    if test_coordinates.shape[0] > 0:
        logging.info(f"Searching the {test_coordinates.shape[0]} remaining coordinates.")
        tail_exp_maps = []
        tail_confidence = []
        for start in range(0, test_coordinates.shape[0], tail_batch_size):
            em, conf, _ = nfutil.search_orientations(
                experiment, controller, image_stack,
                test_coordinates[start:start + tail_batch_size], **search_options)
            tail_exp_maps.append(em)
            tail_confidence.append(conf)
        tail_exp_maps = np.vstack(tail_exp_maps)
        tail_confidence = np.concatenate(tail_confidence)
        valid = tail_confidence > threshold
        # Not grains that are already in the list, from the FF list or the loop above
        new_ones = drop_known_orientations(
            tail_exp_maps[valid], new_exp_maps, 0.5, experiment.plane_data.q_sym)
        logging.info(f"Search done. {valid.sum()} voxels above {threshold} confidence, "
                     f"{valid.sum() - new_ones.sum()} of them already match a grain in the list.")
        logging.info("Merging the rest.")
        final_exp_maps = merge_similar_orientations(
            tail_exp_maps[valid][new_ones], tail_confidence[valid][new_ones], misorientation_deg=0.5,
            crystal_symmetries=experiment.plane_data.q_sym)
    all_exp_maps = np.vstack([new_exp_maps, final_exp_maps])

    save_grains_out(experiment, all_exp_maps)
    if experiment.re_run_and_save == 1:
        rerun_and_save_reconstruction(
            experiment, controller, image_stack, all_exp_maps)

    logging.info(
        f"Total grains found: {all_exp_maps.shape[0]}. Total time: {(time.time() - start_time) / 60.:.2f} minutes.")
    logging.info("Done.")


def find_missing_grains(config_path: str, legacy: bool = False, seed: int = None,
                        batch_size: int = 30, grid_spacing: float = 1.75) -> None:
    """Find missing grains with the orientation search, or the original search if legacy."""
    if legacy:
        find_missing_grains_legacy(config_path, seed=seed)
    else:
        find_missing_grains_search(
            config_path, seed=seed, batch_size=batch_size, grid_spacing=grid_spacing)

# --- CLI Entrypoint ---


def main():
    parser = argparse.ArgumentParser(
        description='Find missing grains in NF reconstruction')
    parser.add_argument('input_file',
                        help='Input configuration file for NF reconstruction')
    parser.add_argument('--legacy', action='store_true',
                        help='Use the original search: simulate the whole ori_grid_spacing grid up front, '
                        'test it at one voxel at a time and refine only the best grid point')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='Voxels searched at once (default search only)')
    parser.add_argument('--grid-spacing', type=float, default=1.75,
                        help='Orientation grid spacing in degrees (default search only; '
                        'the legacy search uses ori_grid_spacing from the config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the random choice of voxels, for repeatable runs')
    args = parser.parse_args()
    find_missing_grains(args.input_file, legacy=args.legacy, seed=args.seed,
                        batch_size=args.batch_size, grid_spacing=args.grid_spacing)


if __name__ == '__main__':
    main()
