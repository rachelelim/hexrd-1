#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:16:10 2022

@author: rachellim

Now with options to decide which grains to look for
"""


# %% #Import Modules
import os
import numpy as np
import copy
import glob

import hexrd.matrixutil as mutil
import hexrd.rotations as rot
import hexrd.symmetry as sym
from hexrd import instrument
from hexrd.cli.fit_grains import write_results
from hexrd import material
from hexrd import valunits

import shutil
import yaml


# %% User Input
###############################################################################

samp = 'inv-diamond-1'
analysis_stem = "%s_" % samp

# hexrd material file in h5 format
mat_file_loc = "../../materials_316L.h5"
mat_name = "316L"
dmin = 0.5  # not really necessary
x_ray_e = 61.332


num_layers = 3
det_names = ['ff1', 'ff2']

load_num = 0


location = 'both'  # generally bottom for tension and top for compression

# 'diamond-1a/FF/%s_S%%d_L%%d/grains_crop.out' % samp
grain_file_stem = '%s_S%%d_L%%d_fixed_grains.out' % samp

# grain_file_stem = '%s_S%%d_L%%d/grains.out' %(samp)
initial_yaml = 'inv-diamond-1_S0.yml'


# %s_%%d_%%s_%%06d-cachefile.npz"%samp
image_fstem = '../../preprocess_dexelas/diamond-1a/diamond-1a_%d_ff1_%06d-cachefile.npz'
first_scan_num = 88
first_file_num = 403
# output_dir_base = "/nfs/chess/user/relim/voisin-3061-c/s26-2"
output_dir_base = './'  # %samp


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


# If your first layer was the bottom layer, these will be negative to positive Y (CHESS and APS default)
pos_offset = np.array(
    [[0., 0., 0.], [-0.475,0., 0.425], [0., 0., 0.]])


beam_height = 0.475

rot_offset = None

low_comp_thresh = 0.5
high_chi2_thresh = 0.05

# vertical dispersion correction
vd_lin = 0.0  # vol_strain/mm
vd_const = 0.0  # vol_strain

# vertical pos correction

v_pos_corr = 0.0

# %%


def quat_multiply(q1, q2):
    """Hamilton product of one quaternion q1 (4,) with quaternions q2 (4, N)

    Quaternions are scalar first, as in hexrd.rotations.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.vstack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def closest_symmetric_variant(q1, q2, qsyms):
    """Symmetry-equivalent of q2 that is closest to q1

    q2 can be written q2*s for any crystal symmetry operator s, so two grains
    with the same physical orientation can have very different quaternions
    (and exponential maps).  Returns that variant (sign chosen so that it is in
    the same hemisphere as q1) and the misorientation angle in degrees.
    """
    variants = quat_multiply(q2, qsyms)
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

    old_grain_numbers = copy.copy(grain_data[:, 0])
    grain_data[:, 0] = np.arange(num_grains)
    return grain_data, old_grain_numbers


def load_pdata(filename, d_min, energy, mat):
    """Note: fixed for new materials files as of hexrd v0.8.x"""

    kev = valunits.valWUnit("beam_energy", "energy", energy, "keV")
    dmin = valunits.valWUnit("dmin", "length", d_min, "angstrom")
    mats = material.load_materials_hdf5(filename, dmin=dmin, kev=kev)
    pd = mats[mat].planeData
    return pd


# makes the lists of grains to check in each layer for timeseries data
def make_grain_lists(analysis_stem, grain_data, state_num, beam_height, pos_offset, output_dir, location='both'):
    vert_off = pos_offset[1]
    gid_offset = np.zeros([len(vert_off)]).astype(int)

    y_loc = grain_data[:, 7]
    print('There are %d grains in the compiled list' % len(grain_data))
    grain_nums = []
    grain_num_fname = 'grain_nums_S%d_L%d.txt'

    ngrains = []

    for i in range(len(vert_off)):

        if location == 'bottom':
            if i == len(vert_off):
                used_grains = np.where(
                    y_loc <= pos_offset[1][i] + beam_height/2 - v_pos_corr)
            elif i == 0:
                used_grains = np.where(
                    y_loc >= pos_offset[1][i] - beam_height*1.5 - v_pos_corr)
            else:
                used_grains = np.where(np.logical_and(
                    y_loc >= pos_offset[1][i]-beam_height*1.5 - v_pos_corr, y_loc <= pos_offset[1][i] + beam_height/2 - v_pos_corr))

        elif location == 'top':
            if i == 0:
                used_grains = np.where(
                    y_loc >= pos_offset[1][i] - beam_height/2 - v_pos_corr)
            elif i == len(vert_off):
                used_grains = np.where(
                    y_loc <= pos_offset[1][i] + beam_height*1.5 - v_pos_corr)

            else:
                used_grains = np.where(np.logical_and(
                    y_loc >= pos_offset[1][i] - beam_height/2, y_loc <= pos_offset[1][i] + beam_height*1.5) - v_pos_corr)
        elif location == 'both':
            if i == len(vert_off):
                used_grains = np.where(
                    y_loc <= pos_offset[1][i] + beam_height*1.5 - v_pos_corr)
            elif i == 0:
                used_grains = np.where(
                    y_loc >= pos_offset[1][i] - beam_height*1.5 - v_pos_corr)
            else:
                used_grains = np.where(np.logical_and(
                    y_loc >= pos_offset[1][i]-beam_height*1.5 - v_pos_corr, y_loc <= pos_offset[1][i] + beam_height*1.5 - v_pos_corr))
                print(pos_offset[1][i]-beam_height*1.5 - v_pos_corr)
                print(pos_offset[1][i] + beam_height*1.5 - v_pos_corr)

        layer_grain_data = grain_data[used_grains[0], :]
        layer_grain_data[:, 7] = layer_grain_data[:, 7] - vert_off[i]
        # print(layer_grain_data)
        # grain_nums.append(used_grains[0])

        np.savetxt(output_dir+grain_num_fname %
                   (state_num, i), used_grains[0].astype(int))

        write_grains_files(output_dir, analysis_stem, layer_grain_data,
                           state_num, layer_num=i, location=location)

    return


def write_grains_files(output_dir, analysis_stem, grain_data, state_num, layer_num=None, location='both'):
    ngrains = len(grain_data)
    if layer_num != None:
        print('\t %d grains in layer %d' % (ngrains, layer_num))
        gids = np.arange(ngrains)
        grain_data[:, 0] = gids

    if location == 'top':
        fstem = "S%d_seed_top_L%d_grains.out"

    if location == 'bottom':
        fstem = "S%d_seed_bottom_L%d_grains.out"

    if location == 'both':
        fstem = "S%d_seed_L%d_grains.out"

    if layer_num == None:
        f = open(os.path.join(output_dir, analysis_stem + "S%03d_master.out" %
                 state_num), "w")
    else:
        f = open(
            os.path.join(
                output_dir,
                fstem % (state_num + 1, layer_num),
            ),
            "w",
        )

    header_items = (
        "grain ID",
        "completeness",
        "chi2",
        "xi[0]",
        "xi[1]",
        "xi[2]",
        "tVec_c[0]",
        "tVec_c[1]",
        "tVec_c[2]",
        "vInv_s[0,0]",
        "vInv_s[1,1]",
        "vInv_s[2,2]",
        "vInv_s[1,2]*sqrt(2)",
        "vInv_s[0,2]*sqrt(2)",
        "vInv_s[0,1]*sqrt(2)",
        "ln(V[0,0])",
        "ln(V[1,1])",
        "ln(V[2,2])",
        "ln(V[1,2])",
        "ln(V[0,2])",
        "ln(V[0,1])",
    )
    len_items = []
    for i in header_items[1:]:
        temp = len(i)
        len_items.append(temp if temp > 19 else 19)  # for %19.12g
    fmtstr = "#%13s  " + "  ".join(["%%%ds" % i for i in len_items]) + "\n"
    f.write(fmtstr % header_items)
    for i in np.arange(grain_data.shape[0]):
        res_items = (
            grain_data[i, 0],
            grain_data[i, 1],
            grain_data[i, 2],
            grain_data[i, 3],
            grain_data[i, 4],
            grain_data[i, 5],
            grain_data[i, 6],
            grain_data[i, 7],
            grain_data[i, 8],
            grain_data[i, 9],
            grain_data[i, 10],
            grain_data[i, 11],
            grain_data[i, 12],
            grain_data[i, 13],
            grain_data[i, 14],
            grain_data[i, 15],
            grain_data[i, 16],
            grain_data[i, 17],
            grain_data[i, 18],
            grain_data[i, 19],
            grain_data[i, 20],
        )
        fmtstr = "%14d  " + \
            "  ".join(["%%%d.12g" % i for i in len_items]) + "\n"
        f.write(fmtstr % res_items)

    f.close()
    return


def write_yaml(init_yaml, image_fstem, analysis_stem, state_num, num_layers, first_scan_num, first_file_num, output_dir, det_keys=['ff1', 'ff2'], location='both'):
    yaml_dict = []
    initial_yaml_info = yaml.load(open(initial_yaml, "r"), Loader=yaml.Loader)
    yaml_dict.append(initial_yaml_info)

    if location == 'top':
        fstem = "S%d_seed_top_L%d_grains.out"
    if location == 'bottom':
        fstem = "S%d_seed_bottom_L%d_grains.out"
    if location == 'both':
        fstem = "S%d_seed_L%d_grains.out"

    for layer_num in range(0, num_layers):
        # this part is a bit derpy but being weird here means the file it creates is cleaner looking
        image_list = []

        # for det_num, det in enumerate(det_names):
        # files = glob.glob(image_fstem % (state_num, layer_num - 1 + first_scan_num, det))

        # for image_file in files:

        # image_file = image_fstem % (state_num, layer_num+111, det)
        for det in det_keys:
            this_dict = {'file': image_fstem % (first_scan_num + layer_num, det, first_file_num + layer_num),
                         'args': {},
                         'panel': ['%s_0_0' % det, '%s_0_1' % det, '%s_1_0' % det, '%s_1_1' % det]}
        # 'file': image_file,
        # 'args': {},
        # 'panel': det}

            image_list.append(this_dict)

        if layer_num == 0:
            initial_yaml_info["analysis_name"] = analysis_stem + \
                "S%d_L%d" % (state_num, layer_num)
            # updating['image_series'] = {}
            initial_yaml_info['image_series']["data"] = image_list
            # updating["fit_grains"] = {}
            initial_yaml_info["fit_grains"]["estimate"] = fstem % (
                state_num, layer_num)

        else:

            updating = {}
            updating["analysis_name"] = analysis_stem + \
                "S%d_L%d" % (state_num, layer_num)
            updating['image_series'] = {}
            updating['image_series']["data"] = image_list
            updating["fit_grains"] = {}
            updating["fit_grains"]["estimate"] = fstem % (state_num, layer_num)

            yaml_dict.append(updating)

    yaml.dump_all(
        yaml_dict,
        open(output_dir + '/' + analysis_stem +
             "S%d.yml" % state_num, "w"),
        default_flow_style=False,
    )

    return  # yaml_dict


# % Load data
pd = load_pdata(mat_file_loc, dmin, x_ray_e, mat_name)
qsyms = pd.q_sym

# %%


# for ll in np.arange(num_states):


output_dir = output_dir_base

num_grain_files = num_layers

grain_data_list = [None] * num_grain_files

for i in np.arange(num_grain_files):
    # analysis_name = analysis_stem + "_S%d_L%d" % (load_num, i+1)
    grain_file = grain_file_stem % (load_num, i)

    # grain_data_list[i]=np.loadtxt(os.path.join(grain_file_locs[i] %(load_num),'grains.out'))
    grain_data_list[i] = np.loadtxt(grain_file)

    pos_0 = grain_data_list[i][:, 6:9]
    grain_data_list[i][:, 15] = grain_data_list[i][:, 15] - (
        vd_lin * pos_0[:, 1] + vd_const
    )
    grain_data_list[i][:, 16] = grain_data_list[i][:, 16] - (
        vd_lin * pos_0[:, 1] + vd_const
    )
    grain_data_list[i][:, 17] = grain_data_list[i][:, 17] - (
        vd_lin * pos_0[:, 1] + vd_const
    )
    grain_data_list[i][:, 7] = grain_data_list[i][:, 7]  # - v_pos_corr

    good_comp = np.where(grain_data_list[i][:, 1] >= low_comp_thresh)[0]
    good_chi2 = np.where(grain_data_list[i][:, 2] <= high_chi2_thresh)[0]

    to_keep = np.intersect1d(good_comp, good_chi2)

    print('Layer %d has %d good grains' % (i, len(to_keep)))

    grain_data_list[i] = grain_data_list[i][to_keep, :]

combined_grain_data, old_grain_numbers = assemble_grain_data(
    grain_data_list, pos_offset, rot_offset
)

#   STUPID POINTERS VS COPYING OF VARIABLES WHY - REL 2022-04-28
combo_gdata = copy.deepcopy(combined_grain_data)

culled_grain_data, dup_list = remove_duplicate_grains(
    combined_grain_data, qsyms, dist, misorientation, completeness_diff
)


new_grain_numbers = np.delete(old_grain_numbers, dup_list.astype(int))


# %%

# make_grain_lists(analysis_stem, culled_grain_data, load_num,
#                  beam_height, pos_offset, output_dir, location=location)
write_grains_files(output_dir, analysis_stem,
                   culled_grain_data, load_num, location=location)
# yaml_dict = write_yaml(initial_yaml, image_fstem, analysis_stem,
#                        load_num+1, num_layers, first_scan_num, first_file_num, output_dir, location=location)

# %%
