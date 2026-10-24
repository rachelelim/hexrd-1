#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Put the layers of the first FF state together into one list of grains

Run this once, on the first state, after calc_lparm.py.  For each layer it reads the grains.out that
hexrd wrote, takes out the energy gradient of the beam, keeps the good grains, moves the layers to their
heights in the sample and merges the grains that are seen in more than one layer.  It writes the master
grains of the state and the seed files and grain number lists of each layer for the next state, which
hexrd starts from.  The hexrd config of the next state is made by build_pipeline.py.

    python stitch_grains.py ff_pipeline.yml [--state 0]

Authors: dcp5303, lim37
"""
import argparse
import copy
import os

import numpy as np

import hexrd.matrixutil as mutil
import hexrd.rotations as rot
from hexrd.core.rotations import quatProduct
from hexrd import material
from hexrd import valunits

from ff_config import load_config
from ff_grains_io import (
    correct_energy_gradient, grains_file_path, load_beam_calibration,
    make_grain_lists, read_beam_energy, write_grains_files,
)


def closest_symmetric_variant(q1, q2, qsyms):
    """Symmetry-equivalent of q2 that is closest to q1

    q2 can be written q2*s for any crystal symmetry operator s, so two grains
    with the same physical orientation can have very different quaternions
    (and exponential maps).  Returns that variant (sign chosen so that it is in
    the same hemisphere as q1) and the misorientation angle in degrees.
    """
    # q2 * s for every symmetry operator s; quatProduct(a, b) is b * a
    variants = quatProduct(qsyms, q2[:, None])
    dots = q1 @ variants
    best = np.argmax(np.abs(dots))
    q2_best = variants[:, best] * np.sign(dots[best])
    angle = 2.0 * np.arccos(np.clip(np.abs(dots[best]), 0.0, 1.0))
    return q2_best, np.degrees(angle)


def remove_duplicate_grains(
    grain_data, qsyms, dist_thresh=0.01, misorient_thresh=0.1, comp_diff=0.1
):
    # grains.out columns: 3:6 exponential map, 6:9 centroid (tVec_c)
    total_grains = grain_data.shape[0]

    all_comp = grain_data[:, 1]
    grain_quats = rot.quatOfExpMap(grain_data[:, 3:6].T)
    dup_list = np.array([])

    print("Removing duplicate grains")
    for i in np.arange(total_grains - 1):
        cur_pos = grain_data[i, 6:9]
        other_pos = grain_data[(i + 1):, 6:9]
        xdist = cur_pos[0] - other_pos[:, 0]
        ydist = cur_pos[1] - other_pos[:, 1]
        zdist = cur_pos[2] - other_pos[:, 2]

        dist = np.sqrt(xdist ** 2.0 + ydist ** 2.0 + zdist ** 2.0)

        nearest = np.argmin(dist) + i + 1

        if np.min(dist) < dist_thresh:

            q2_aligned, misorientation = closest_symmetric_variant(
                grain_quats[:, i], grain_quats[:, nearest], qsyms
            )

            if misorientation < misorient_thresh:
                dup_list = np.append(dup_list, nearest)
                if np.abs(all_comp[i] - all_comp[nearest]) <= comp_diff:
                    merged = (grain_data[i, :] + grain_data[nearest, :]) / 2.0
                    # exponential maps of symmetry-equivalent orientations can
                    # differ wildly, so average the aligned quaternions instead
                    q_avg = grain_quats[:, i] + q2_aligned
                    q_avg = q_avg / np.linalg.norm(q_avg)
                    merged[3:6] = rot.expMapOfQuat(
                        np.atleast_2d(q_avg).T).flatten()
                    grain_data[i, :] = merged
                elif (all_comp[nearest] - all_comp[i]) > 0:
                    grain_data[i, :] = grain_data[nearest, :]

    grain_data = np.delete(grain_data, dup_list.astype(int), axis=0)

    print("%d Duplicate Grains" % (len(dup_list)))

    grain_data[:, 0] = np.arange(grain_data.shape[0])

    return grain_data, dup_list


def assemble_grain_data(grain_data_list, pos_offset=None, rotation_offset=None):
    num_grain_files = len(grain_data_list)

    num_grains_list = [None] * num_grain_files

    for i in np.arange(num_grain_files):
        num_grains_list[i] = grain_data_list[i].shape[0]

    num_grains = np.sum(num_grains_list)

    grain_data = np.zeros([num_grains, grain_data_list[0].shape[1]])

    for i in np.arange(num_grain_files):

        tmp = copy.copy(grain_data_list[i])

        if pos_offset is not None:
            pos_tile = np.tile(pos_offset[:, i], [num_grains_list[i], 1])
            tmp[:, 6:9] = tmp[:, 6:9] + pos_tile
        # Needs Testing
        if rotation_offset is not None:
            rot_tile = np.tile(
                np.atleast_2d(rotation_offset[:, i]).T, [1, num_grains_list[i]]
            )
            quat_tile = rot.quatOfExpMap(rot_tile)
            grain_quats = rot.quatOfExpMap(tmp[:, 3:6].T)
            new_quats = rot.quatProduct(grain_quats, quat_tile)

            sinang = mutil.columnNorm(new_quats[1:, :])
            ang = 2.0 * np.arcsin(sinang)
            axis = mutil.unitVector(new_quats[1:, :])
            tmp[:, 3:6] = np.tile(np.atleast_2d(ang).T, [1, 3]) * axis.T

        grain_data[
            int(np.sum(num_grains_list[:i])): int(np.sum(num_grains_list[: (i + 1)])),
            :,
        ] = tmp

    grain_data[:, 0] = np.arange(num_grains)
    return grain_data


def load_pdata(filename, energy, mat):
    """Plane data of a material, at the beam energy and with the dmin saved in the materials file

    Note: fixed for new materials files as of hexrd v0.8.x"""

    kev = valunits.valWUnit("beam_energy", "energy", energy, "keV")
    mats = material.load_materials_hdf5(filename, kev=kev)
    print('Material %s: beam energy %g keV, dmin %s' % (mat, energy, mats[mat].dmin))
    pd = mats[mat].planeData
    return pd


def stitch(cfg, state=0):
    """Make the master grains and the seed files of the layers of a state, from the config

    cfg is from ff_config.load_config.  Returns the master grains, with the duplicates merged.
    """
    s = cfg.stitch
    qsyms = load_pdata(cfg.materials_file, read_beam_energy(cfg.detector_file), cfg.material_name).q_sym

    beam_calibration = None
    beam_center_y = 0.0
    if cfg.beam_calibration_file is not None:
        beam_calibration = load_beam_calibration(cfg.beam_calibration_file)
        beam_center_y = beam_calibration['beam_center_y']

    grain_data_list = []
    for i in range(cfg.num_layers):
        data = np.loadtxt(grains_file_path(cfg.ff_dir, cfg.analysis_stem, state, i))
        # the y of the grains is still in the frame of the layer here, as the gradient is
        if beam_calibration is not None:
            data = correct_energy_gradient(data, beam_calibration)
        good = np.logical_and(data[:, 1] >= s.low_comp_thresh, data[:, 2] <= s.high_chi2_thresh)
        print('Layer %d has %d good grains' % (i, good.sum()))
        grain_data_list.append(data[good, :])

    combined_grain_data = assemble_grain_data(grain_data_list, cfg.pos_offset, None)
    culled_grain_data, dup_list = remove_duplicate_grains(
        combined_grain_data, qsyms, s.dist, s.misorientation, s.completeness_diff
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    write_grains_files(cfg.output_dir, cfg.analysis_stem, culled_grain_data, state, location=cfg.location)
    if s.make_layer_grain_lists:
        make_grain_lists(cfg.analysis_stem, culled_grain_data, state, cfg.beam_height, cfg.pos_offset,
                         cfg.output_dir, location=cfg.location, v_pos_corr=s.v_pos_corr,
                         beam_center_y=beam_center_y, seed_margin=cfg.seed_margin)
    return culled_grain_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('config', help='the pipeline config yaml')
    parser.add_argument('--state', type=int, default=0, help='the state to put together, 0 for the first')
    args = parser.parse_args(argv)
    stitch(load_config(args.config), args.state)


if __name__ == '__main__':
    main()
