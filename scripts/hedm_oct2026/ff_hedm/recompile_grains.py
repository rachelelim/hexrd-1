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

Run this for every state after the first, once hexrd has fit the grains of every layer of the state,
that is, once <analysis_stem>S<state>_L<layer>/grains.out exist in the output directory.  It puts the
layers back together into one grains list, then writes the files for the next state in the same form
as stitch_grains.py, which is only run for the first state:

    <stem>S<state>_master.out            the compiled grains
    grain_nums_S<state>_L<layer>.txt     which grains are in each layer
    S<state+1>_seed_L<layer>_grains.out  the starting grains of each layer

The hexrd config of the next state is made by build_pipeline.py.

    python recompile_grains.py ff_pipeline.yml --state 3

Authors: dcp5303, lim37
"""
import argparse
import os

import numpy as np

from ff_config import load_config
from ff_grains_io import (
    correct_energy_gradient, grain_nums_file_name, grains_file_path,
    load_beam_calibration, make_grain_lists, master_file_name, write_grains_files,
)


def recompile(cfg, state):
    """Compile the master grains of a state from the fits of its layers, and seed the next state

    cfg is from ff_config.load_config.  Returns the master grains.
    """
    if state < 1:
        raise ValueError('The first state (0) is put together by stitch_grains.py, not recompiled')
    r = cfg.recompile
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # The compiled grains of the last state, which seeded the layers that were just fit
    master_grain_data = np.loadtxt(os.path.join(output_dir, master_file_name(cfg.analysis_stem, state - 1)))
    num_grains = master_grain_data.shape[0]

    beam_calibration = None
    beam_center_y = 0.0
    if cfg.beam_calibration_file is not None:
        beam_calibration = load_beam_calibration(cfg.beam_calibration_file)
        beam_center_y = beam_calibration['beam_center_y']

    # The fit of each layer, in the frame of the sample, in the rows of the master list
    grain_holder = np.zeros([cfg.num_layers, num_grains, master_grain_data.shape[1]])

    for i in range(cfg.num_layers):
        scan_grain_data = np.loadtxt(grains_file_path(output_dir, cfg.analysis_stem, state, i))
        grain_num_list = np.loadtxt(os.path.join(
            output_dir, grain_nums_file_name(state - 1, i))).astype(int)

        print('Layer %d: %d grains fit, %d with completeness above 0.9'
              % (i, len(scan_grain_data), np.sum(scan_grain_data[:, 1] > 0.9)))

        # the y of the grains is still in the frame of the layer here, as the gradient is
        if beam_calibration is not None:
            scan_grain_data = correct_energy_gradient(scan_grain_data, beam_calibration)

        scan_grain_data[:, 6:9] = scan_grain_data[:, 6:9] + cfg.pos_offset[:, i]

        grain_holder[i, grain_num_list, :] = scan_grain_data

    # Compile one row for each grain from its layers
    grain_data = np.zeros([num_grains, 21])
    for kk in range(num_grains):
        max_comp = np.max(grain_holder[:, kk, 1])
        # layers whose completeness is close to the best one
        close_vals = np.where(
            np.abs(grain_holder[:, kk, 1] - max_comp) < r.close_completeness_cutoff)[0]
        chi2_vals = grain_holder[:, kk, 2]
        good_grains = np.where(chi2_vals < r.high_chi2_thresh)[0]
        test = np.intersect1d(close_vals, good_grains)

        if len(good_grains) <= 1:
            best_grain = np.argmin(chi2_vals)
            grain_data[kk, 1:] = grain_holder[best_grain, kk, 1:]
        else:
            grain_data[kk, 1:] = np.mean(grain_holder[test, kk, 1:], 0)

    grain_data[:, 0] = np.arange(num_grains)

    print('%d grains in %d layers, %d with completeness above 0.9, %d grains disappeared'
          % (num_grains, cfg.num_layers, np.sum(grain_data[:, 1] > 0.9), np.sum(grain_data[:, 1] == 0)))

    # The master list and the files for the next state
    write_grains_files(output_dir, cfg.analysis_stem, grain_data, state, location=cfg.location)
    make_grain_lists(cfg.analysis_stem, grain_data, state, cfg.beam_height, cfg.pos_offset, output_dir,
                     location=cfg.location, low_comp_thresh=r.low_comp_thresh,
                     high_chi2_thresh=r.high_chi2_thresh, beam_center_y=beam_center_y,
                     seed_margin=cfg.seed_margin)
    return grain_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('config', help='the pipeline config yaml')
    parser.add_argument('--state', type=int, required=True,
                        help='the state that hexrd has just fit, 1 or later')
    args = parser.parse_args(argv)
    recompile(load_config(args.config), args.state)


if __name__ == '__main__':
    main()
