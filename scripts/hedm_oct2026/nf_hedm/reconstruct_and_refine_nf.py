#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
original author: dcp5303
contributing author: seg246
"""
"""
NOTES:
    - The reference frame used in this script is the HEXRD frame
        - X points right if facing towards the x-ray detector (downstream)
        - Y points up (against gravity)
        - Z points upstream (away from the detector)
        - X,Y,Z center is where the beam intercepts the rotation axis

    - The tomography mask input is a binarized array of the entire tomography volume
    - Y (vertical) calibration is only needed if your detector center was not
        placed at the center of the beam (usually the case)
    - Z distance is around 6-7 mm normally (11 mm if the furnace is in)
    - X distance is the same as from your tomography reconstruction
        If you have a RAMS sample then it is usually less than 0.1 mm
    - You voxel size should not be less than your pixel size - you cannot claim such resolution
    - The IPF color plotting has not be unit tested and as such it should not be used for 
        anything but general debugging and initial visualization (currently)
    - This reconstruction alogorithm only produces a grain averaged microstructure
        If your sample has high dislocation content it will not do well
    - For choosing the HKLs, it is advised to draw out the rough geometry and run 
        the numbers to see which HKLs will hit the detector at the front and 
        back of the sample - refine as you calibration z
    - If FF did not find a grain, it will show up as a low confidence region in
        the NF reconstruction
    - Your images have already been processed and binarized with a prior script


    - Note that HEXRD works with grain orientations which are defined from CRYSTAL TO SAMPLE 
        specifically as v_samp = R_cry_to_samp * v_cry
    - Here is a description pulled from the xf.py script within HEXRD.  (COB is change of basis). 
    
        gVec_c : numpy.ndarray
            (3, n) array of n reciprocal lattice vectors in the CRYSTAL FRAME.
        rMat_s : numpy.ndarray
            (3, 3) array, the COB taking SAMPLE FRAME components to LAB FRAME. 
        rMat_c : numpy.ndarray
            (3, 3) array, the COB taking CRYSTAL FRAME components to SAMPLE FRAME.
    
        # form unit reciprocal lattice vectors in lab frame (w/o translation)
        gVec_l = np.dot(rMat_s, np.dot(rMat_c, unitVector(gVec_c)))
    
    The line above is the important one.  It is the coordinate transformation of the g vector 
        from the crystal frame to the lab frame.  rMat_c is the orientation matrix that we have 
        outputted into the grain.out files from HEXRD (though it is expressed in axis angle form).  
        Note that rMat_c is defined from the crystal to the sample frame, but more specifically it transforms a 
        vector in the crystal frame into the sample frame as: gVec_s = np.dot(rMat_c,gVec_c).  Note that 
        np.dot in python is the same as rMat_c*gVec_c in matlab (a pre-multiplication).  Recall 
        of course that gVec_c here is a column vector (tall – 3x1 in both python and matlab) as is gVec_s.

"""

# %% ===========================================================================
# Imports - NO CHANGES NEEDED
# ==============================================================================
# General Imports
from hexrd.hedm import nf_config
import os
import sys

# nfutil lives in the parent (hedm_oct2026) folder
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nfutil
import argparse

# Hexrd imports

# Matplotlib
# This is to allow interactivity of inline plots in your gui
# the import ipywidgets as widgets line is not needed - however, you do need to run a pip install ipywidgets
# the import ipympl line is not needed - however, you do need to run a pip install ipympl
# import ipywidgets as widgets
# import ipympl
# The next lines are formatted correctly, no matter what your IDE says
# For inline, interactive plots (if you use these, make sure to run a plt.close() to prevent crashing)
# %matplotlib widget
# For inline, non-interactive plots
# %matplotlib inline
# For pop out, interactive plots (cannot be used with an SSH tunnel)
# %matplotlib qt

# %% ==============================================================================
# FILES TO LOAD -CAN BE EDITED
# ==============================================================================

parser = argparse.ArgumentParser(description='Reconstruct NF microstructure with orientation refinement')

parser.add_argument('input_file',
                    help='Input file for NF reconstruction')
parser.add_argument('--spatially-adaptive', action='store_true',
                    help='Refine each voxel starting from its neighbors\' refined orientations '
                    'instead of from its grain\'s FF orientation. About 5 times faster, follows '
                    'orientation gradients inside grains, and gives about the same confidence')
parser.add_argument('--converged-confidence', type=float, default=0.7,
                    help='Confidence above which a voxel seeds its neighbors')
parser.add_argument('--drop-off', type=float, default=0.8,
                    help='A neighbor is accepted if its confidence is at least '
                    'this fraction of the voxel it starts from')
parser.add_argument('--seed-spacing', type=int, default=8,
                    help='Voxel spacing of the seed grid, whose voxels are tested against every FF grain')


args = parser.parse_args()
configuration_filepath = args.input_file
# configuration_filepath = 'nf_config_S0_L0.yml'

# %% ==========================================================================
# LOAD IMAGES AND EXPERIMENT - DO NOT EDIT
# =============================================================================
# Go ahead and load the configuration
configuration = nf_config.open_file(configuration_filepath)[0]
# Generate the experiment
experiment = nfutil.Experiment(configuration)
image_stack = experiment.image_stack
# Generate the controller
controller = nfutil.build_controller(configuration)
# %% ===========================================================================
# LOAD MASK / GENERATE TEST COORDINATES  - NO CHANGES NEEDED
# ==============================================================================
Xs, Ys, Zs, mask, test_coordinates = experiment.generate_test_coordinates()
# %% ==========================================================================
# PRECOMPUTE ORIENTATION DATA
# =============================================================================
precomputed_orientation_data = nfutil.precompute_diffraction_data(
    experiment, controller, experiment.exp_maps)

# %% ==========================================================================
# TEST ORIENTATIONS AND PROCESS OUTPUT
# =============================================================================
if args.spatially_adaptive:
    raw_exp_maps, raw_confidence, raw_idx, raw_misorientation, stats = nfutil.reconstruct_and_refine_spatially_adaptive(
        experiment, controller, image_stack, precomputed_orientation_data, test_coordinates, mask,
        converged_confidence=args.converged_confidence, drop_off=args.drop_off,
        seed_spacing=args.seed_spacing)
    print(f"Refined along the marching: {stats['n_local']} voxels started from their neighbors, "
          f"{stats['n_seed'] + stats['n_global']} from the best FF grain; "
          f"{stats['n_simulated']} orientations simulated "
          f"({stats['n_simulated']/stats['n_voxels']:.0f} per voxel).")
else:
    raw_exp_maps, raw_confidence, raw_idx, raw_misorientation = nfutil.test_orientations_at_coordinates(
        experiment, controller, image_stack, precomputed_orientation_data, test_coordinates, refine_yes_no=1, return_misorientation=1)
grain_map, confidence_map, misorientation_map = nfutil.process_raw_data(
    raw_confidence, raw_idx, Xs.shape, mask=mask, id_remap=experiment.remap, raw_misorientation=raw_misorientation)

# %% ==========================================================================
# Show Images - CAN BE EDITED
# =============================================================================
if configuration.output_plot_check is True:
    layer_num = 0  # Which layer in Y?
    conf_thresh = 0.2  # If set to None no threshold is used
    nfutil.plot_ori_map(grain_map, confidence_map, Xs, Zs, experiment.exp_maps,
                        layer_num, experiment.mat[experiment.material_name],
                        experiment.remap, conf_thresh, misorientation_map=misorientation_map)
# Quick note - nfutil assumes that the IPF reference vector is [0 1 0]

# %% ==========================================================================
# SAVE PROCESSED GRAIN MAP DATA - CAN BE EDITED
# =============================================================================
nfutil.save_nf_data(experiment.output_directory, experiment.analysis_name, grain_map, confidence_map,
                    Xs, Ys, Zs, experiment.exp_maps, tomo_mask=mask, id_remap=experiment.remap,
                    save_type=['npz'])  # Can be npz or hdf5

# %% ==========================================================================
# SAVE PROCESSED GRAIN MAP DATA WITH IPF COLORS - CAN BE EDITED
# =============================================================================
nfutil.save_nf_data_for_paraview(experiment.output_directory, experiment.analysis_name, grain_map, confidence_map, Xs, Ys, Zs,
                                 experiment.exp_maps, experiment.mat[experiment.material_name], tomo_mask=mask,
                                 id_remap=experiment.remap, misorientation_map=misorientation_map)
# Quick note - nfutil assumes that the IPF reference vector is [0 1 0]


# %%
