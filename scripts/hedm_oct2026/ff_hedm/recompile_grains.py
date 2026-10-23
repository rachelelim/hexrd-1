# ============================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on downloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# ============================================================
###############################################################################
"""Recompile the FF grains of a state and set up the next one

Run this after hexrd has fit the grains of every layer of state load_num, i.e.
after <analysis_stem>S<load_num>_L<layer>/grains.out exist.  It puts the layers
back together into one grains list, then writes the files for state
load_num + 1 in the same form as stitch_grains.py, which is only
run for the first state:

    <stem>S<load_num>_master.out        compiled grains
    grain_nums_S<load_num>_L<layer>.txt  which grains are in each layer
    S<load_num+1>_seed_L<layer>_grains.out  starting grains of each layer
    <stem>S<load_num+1>.yml             hexrd config for the next state (optional)

The settings that describe the experiment (sample, layers, offsets, beam height,
detector file, image names) have to match the ones in stitch_grains.py.
"""

# %% Import Modules
import os

import numpy as np

from ff_grains_io import (
    analysis_name, grain_nums_file_name, make_grain_lists, master_file_name,
    read_detector_panels, write_grains_files, write_yaml,
)


# %% User Input
###############################################################################

samp = 'diamond-1a'
analysis_stem = "%s_" % samp

# State that hexrd has just fit; the grains it started from are in state load_num - 1
load_num = 1
num_layers = 4
location = 'both'  # as in the stitching script

# Directory with the master and grain_nums files and the hexrd analysis directories
output_dir_base = './'

data_dir = "/Users/lim37/Library/CloudStorage/OneDrive-LLNL/TriMet - General/CHESS_2026-09"
# hexrd instrument file; its detector keys and their panels are used for the yaml image series
detector_file = data_dir + '/dexela_ruby_composite.hexrd'

# Reassembly
close_completness_cutoff = 0.1  # 10% is probably reasonable
high_chi2_thresh = 5e-2

# Seeding the next state
low_comp_thresh = 0.6

# Vertical (ramsz) motor position of each layer, Y offset is -ramsz, as in the stitching script
ramsz = np.array([-0.525, -0.125, 0.125, 0.525])
pos_offset = np.zeros([3, len(ramsz)])
pos_offset[1, :] = -ramsz

beam_height = 0.35

# Optional FF config yaml for the next state, built from initial_yaml and image_fstem
write_next_yaml = False

initial_yaml = 'diamond-1a_S0.yml'
# image files per scan, filled with (sample, scan number, detector name, file number)
image_fstem = '../../preprocess_dexelas/diamond-1a/%s_%d_%s_%06d-cachefile.npz'
# of the first layer of the next state
first_scan_num = 88
first_file_num = 403


# %% Layer grains to the master list
###############################################################################

output_dir = output_dir_base
os.makedirs(output_dir, exist_ok=True)

# The compiled grains of the last state, which seeded the layers that were just fit
master_grain_data = np.loadtxt(os.path.join(
    output_dir, master_file_name(analysis_stem, load_num - 1)))
num_grains = master_grain_data.shape[0]

# The fit of each layer, in the frame of the sample, in the rows of the master list
grain_holder = np.zeros([num_layers, num_grains, master_grain_data.shape[1]])

for i in np.arange(num_layers):
    scan_grain_data = np.loadtxt(os.path.join(
        output_dir, analysis_name(analysis_stem, load_num, i), 'grains.out'))
    grain_num_list = np.loadtxt(os.path.join(
        output_dir, grain_nums_file_name(load_num - 1, i))).astype(int)

    print('Layer %d: %d grains fit, %d with completeness above 0.9'
          % (i, len(scan_grain_data), np.sum(scan_grain_data[:, 1] > 0.9)))

    scan_grain_data[:, 6:9] = scan_grain_data[:, 6:9] + pos_offset[:, i]

    grain_holder[i, grain_num_list, :] = scan_grain_data


# %% Compile the master list
###############################################################################

grain_data = np.zeros([num_grains, 21])
n_good = 0

for kk in np.arange(num_grains):

    max_comp = np.max(grain_holder[:, kk, 1])
    # layers whose completeness is close to the best one
    close_vals = np.where(
        np.abs(grain_holder[:, kk, 1] - max_comp) < close_completness_cutoff)[0]
    chi2_vals = grain_holder[:, kk, 2]
    good_grains = np.where(chi2_vals < high_chi2_thresh)[0]
    test = np.intersect1d(close_vals, good_grains)

    if len(good_grains) <= 1:
        best_grain = np.argmin(chi2_vals)
        grain_data[kk, 1:] = grain_holder[best_grain, kk, 1:]
        n_good += 1
    else:
        grain_data[kk, 1:] = np.mean(grain_holder[test, kk, 1:], 0)

grain_data[:, 0] = np.arange(num_grains)

print('%d grains in %d layers, %d with completeness above 0.9, %d grains disappeared'
      % (num_grains, num_layers, np.sum(grain_data[:, 1] > 0.9), np.sum(grain_data[:, 1] == 0)))


# %% Write the master list and the files for the next state
###############################################################################

write_grains_files(output_dir, analysis_stem, grain_data, load_num, location=location)

make_grain_lists(analysis_stem, grain_data, load_num, beam_height, pos_offset, output_dir,
                 location=location, low_comp_thresh=low_comp_thresh,
                 high_chi2_thresh=high_chi2_thresh)

if write_next_yaml:
    write_yaml(initial_yaml, image_fstem, samp, analysis_stem, load_num + 1,
               num_layers, first_scan_num, first_file_num, output_dir,
               det_panels=read_detector_panels(detector_file), location=location)
