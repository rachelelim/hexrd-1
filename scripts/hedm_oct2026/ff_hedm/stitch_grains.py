#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:16:10 2022

@author: rachellim

Now with options to decide which grains to look for
"""


# %% #Import Modules
import copy

import numpy as np

import hexrd.matrixutil as mutil
import hexrd.rotations as rot
from hexrd.core.rotations import quatProduct
from hexrd import material
from hexrd import valunits

from ff_grains_io import (
    beam_calibration_file_name, correct_energy_gradient, load_beam_calibration,
    make_grain_lists, read_beam_energy, read_detector_panels,
    write_grains_files, write_yaml,
)


# %% User Input
###############################################################################

samp = 'diamond-1a'
analysis_stem = "%s_" % samp

# hexrd material file in h5 format
data_dir = "/Users/lim37/Library/CloudStorage/OneDrive-LLNL/TriMet - General/CHESS_2026-09"
mat_file_loc = data_dir + "/materials_316L.h5"
mat_name = "316L"
# the beam energy comes from detector_file and dmin from the materials file


num_layers = 4
# hexrd instrument file; its detector keys and their panels are used for the yaml image series
detector_file = data_dir + '/dexela_ruby_composite.hexrd'

load_num = 0


# Where the seed grains of each layer are taken from: the volume that the beam illuminates (the
# beam height around the beam center) and a margin beyond it, which takes in grains from just
# outside that could move into it.  y is up in the lab frame.
#   'both'  - the margin is above and below the illuminated volume
#   'above' - the margin is only above it, toward higher y
#   'below' - the margin is only below it, toward lower y
# (The old names were 'top' for 'above' and 'bottom' for 'below', with the note that it is
# generally below for tension and above for compression.)
location = 'both'
# Size of that margin, as a fraction of the beam height
seed_margin = 0.5

grain_file_stem = data_dir + '/%s/FF/%s_S%%d_L%%d_grains.out' % (samp, samp)

# Optional outputs for the next state
# Per-layer seed grains.out files (S<next>_seed_L<layer>_grains.out) and the grain_nums_S*_L*.txt lists
make_layer_grain_lists = True
# FF config yaml for the next state, built from initial_yaml and image_fstem
write_next_yaml = False

initial_yaml = 'diamond-1a_S0.yml'
# image files per scan, filled with (sample, scan number, detector name, file number)
image_fstem = '../../preprocess_dexelas/diamond-1a/%s_%d_%s_%06d-cachefile.npz'
first_scan_num = 88
first_file_num = 403
output_dir_base = './'


# Position and Misorientation differences to merge grains
dist = 0.1  # mm
misorientation = 0.25  # degrees
completeness_diff = 0.1  # if two grains are matched, the completenesses are checked,
# if the differences in completion are within completeness_diff, the grain values are averaged,
# if not, the data from the grain with higher completion is kept and the other data is discarded

# Offsets, these can be input as arguments
# Each dataset can have positional or rotation offsets
# Position offsets are in mm, 3 x n matrix where n is the number of grains.out files being stitched
# Rotation offsets exponential maps


# Vertical (ramsz) motor position of each layer. Layers lower in the sample have a higher ramsz
# (the motor sits below the sample), so the Y offset applied to each grains.out is -ramsz
ramsz = np.array([-0.525, -0.125, 0.125, 0.525])
pos_offset = np.zeros([3, len(ramsz)])
pos_offset[1, :] = -ramsz

beam_height = 0.35

rot_offset = None

low_comp_thresh = 0.5
high_chi2_thresh = 0.05

# Energy gradient of the beam, to correct the strains for, and the height of the center of the
# beam, which calc_lparm.py finds (None for no correction and a beam center at 0)
beam_calibration_file = output_dir_base + beam_calibration_file_name(analysis_stem)

# vertical pos correction

v_pos_corr = 0.0

# %%


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


# % Load data
x_ray_e = read_beam_energy(detector_file)
pd = load_pdata(mat_file_loc, x_ray_e, mat_name)
qsyms = pd.q_sym

# %%


# for ll in np.arange(num_states):


output_dir = output_dir_base

num_grain_files = num_layers

grain_data_list = [None] * num_grain_files

beam_calibration = None
beam_center_y = 0.0
if beam_calibration_file is not None:
    beam_calibration = load_beam_calibration(beam_calibration_file)
    beam_center_y = beam_calibration['beam_center_y']

for i in np.arange(num_grain_files):
    # analysis_name = analysis_stem + "_S%d_L%d" % (load_num, i+1)
    grain_file = grain_file_stem % (load_num, i)

    # grain_data_list[i]=np.loadtxt(os.path.join(grain_file_locs[i] %(load_num),'grains.out'))
    grain_data_list[i] = np.loadtxt(grain_file)

    # the y of the grains is still in the frame of the layer here, as the gradient is
    if beam_calibration is not None:
        grain_data_list[i] = correct_energy_gradient(grain_data_list[i], beam_calibration)

    good_comp = np.where(grain_data_list[i][:, 1] >= low_comp_thresh)[0]
    good_chi2 = np.where(grain_data_list[i][:, 2] <= high_chi2_thresh)[0]

    to_keep = np.intersect1d(good_comp, good_chi2)

    print('Layer %d has %d good grains' % (i, len(to_keep)))

    grain_data_list[i] = grain_data_list[i][to_keep, :]

combined_grain_data = assemble_grain_data(
    grain_data_list, pos_offset, rot_offset
)

culled_grain_data, dup_list = remove_duplicate_grains(
    combined_grain_data, qsyms, dist, misorientation, completeness_diff
)


# %%

write_grains_files(output_dir, analysis_stem,
                   culled_grain_data, load_num, location=location)

if make_layer_grain_lists:
    make_grain_lists(analysis_stem, culled_grain_data, load_num,
                     beam_height, pos_offset, output_dir, location=location,
                     v_pos_corr=v_pos_corr, beam_center_y=beam_center_y,
                     seed_margin=seed_margin)

if write_next_yaml:
    write_yaml(initial_yaml, image_fstem, samp, analysis_stem, load_num + 1,
               num_layers, first_scan_num, first_file_num, output_dir,
               det_panels=read_detector_panels(detector_file), location=location)

# %%
