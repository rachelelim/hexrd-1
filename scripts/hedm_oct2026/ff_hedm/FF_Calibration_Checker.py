
"""
Author: seg246, jward
"""
# %% =====================================
# DEPENDENCIES AND FUNCTIONS - DO NOT EDIT
# =======================================
import yaml
import numpy as np
import os
import h5py
from hexrd import instrument
from matplotlib import pyplot as plt


def generate_cal_check_plots(output_filepath, instrument_file, grains_out_directory_path, index_to_plot, detector, histogram_nbins=50, translation_bounds=0.2, tth_bounds=0.13, eta_bounds=0.2, ome_bounds=0.5):
    #  ======================================
    # LOADING DATA - DO NOT EDIT
    # =======================================
    file_name, file_extension = os.path.splitext(instrument_file)
    if file_extension == '.yml':
        with open(instrument_file, 'r') as f:
            icfg = yaml.load(f, Loader=yaml.SafeLoader)
        instr = instrument.HEDMInstrument(instrument_config=icfg)
    elif file_extension == '.hexrd':
        with h5py.File(instrument_file, 'r') as rf:
            instr = instrument.HEDMInstrument(rf)

    grains_out_file = np.loadtxt(grains_out_directory_path+'grains.out')
    num_grains = np.shape(grains_out_file)[0]
    num_spots = 0

    if detector == 'dexela_subpanels':
        num_panels = 8
        all_data = np.empty([num_grains, num_panels], object)
        folder_names = np.array(['ff1_0_0', 'ff1_0_1', 'ff1_1_0',
                                'ff1_1_1', 'ff2_0_0', 'ff2_0_1', 'ff2_1_0', 'ff2_1_1'])
    elif detector == 'dexela_monoliths':
        num_panels = 2
        all_data = np.empty([num_grains, num_panels], object)
        folder_names = np.array(['ff1', 'ff2'])
    elif detector == 'eiger':
        num_panels = 1
        all_data = np.empty([num_grains, num_panels], object)
        folder_names = np.array(['eiger'])
    elif detector == 'eiger_partial':
        num_panels = 4
        all_data = np.empty([num_grains, num_panels], object)
        folder_names = np.array(['eiger_0_0',
                                 'eiger_0_1',
                                 'eiger_0_2',
                                 'eiger_0_3',])
    elif detector == 'eiger_subpanels':
        num_panels = 32
        all_data = np.empty([num_grains, num_panels], object)
        folder_names = np.array(['eiger_0_0',
                                 'eiger_0_1',
                                 'eiger_0_2',
                                 'eiger_0_3',
                                 'eiger_1_0',
                                 'eiger_1_1',
                                 'eiger_1_2',
                                 'eiger_1_3',
                                 'eiger_2_0',
                                 'eiger_2_1',
                                 'eiger_2_2',
                                 'eiger_2_3',
                                 'eiger_3_0',
                                 'eiger_3_1',
                                 'eiger_3_2',
                                 'eiger_3_3',
                                 'eiger_4_0',
                                 'eiger_4_1',
                                 'eiger_4_2',
                                 'eiger_4_3',
                                 'eiger_5_0',
                                 'eiger_5_1',
                                 'eiger_5_2',
                                 'eiger_5_3',
                                 'eiger_6_0',
                                 'eiger_6_1',
                                 'eiger_6_2',
                                 'eiger_6_3',
                                 'eiger_7_0',
                                 'eiger_7_1',
                                 'eiger_7_2',
                                 'eiger_7_3',
                                 ])
    else:
        print(f'Detector {detector} is not implemented.')

    num_spots = 0
    for grain in np.arange(num_grains):
        for file in np.arange(num_panels):
            filepath = grains_out_directory_path + '/' + \
                folder_names[file] + '/' + 'spots_' + \
                f'{grain}'.zfill(5) + '.out'
            print(filepath)
            data = np.loadtxt(filepath, skiprows=0)
            spots_to_remove = data[:, 0] > -1
            all_data[grain, file] = data[spots_to_remove, :]
            num_spots = num_spots + np.shape(all_data[grain, file])[0]

    predicted_values = np.zeros([num_spots, 5])
    measured_values = np.zeros([num_spots, 5])

    current_spot = 0
    for grain in np.arange(num_grains):
        for panel in np.arange(num_panels):
            start = current_spot
            stop = current_spot + np.shape(all_data[grain, panel])[0]
            predicted_values[start:stop] = all_data[grain,
                                                    panel][:, [7, 8, 9, 13, 14]]
            measured_values[start:stop] = all_data[grain,
                                                   panel][:, [10, 11, 12, 15, 16]]
            predicted_values[start:stop, 3:] = predicted_values[start:stop, 3:] + \
                instr.detector_parameters[folder_names[panel]
                                          ]['detector']['transform']['translation'][0:2]
            measured_values[start:stop, 3:] = measured_values[start:stop, 3:] + \
                instr.detector_parameters[folder_names[panel]
                                          ]['detector']['transform']['translation'][0:2]

            current_spot = stop

    x = predicted_values[:, 3]
    y = predicted_values[:, 4]

    # =====================================
    # MAKING OUTPUT DIRECTORY - DO NOT EDIT
    if output_filepath is not None:
        if not os.path.exists(output_filepath):
            os.mkdir(output_filepath)

    # ===================================
    # PLOTTING - DO NOT EDIT
    # =====================================

    if index_to_plot == 0:  # Tth
        sim_val = predicted_values[:, 0] * 180/np.pi
        exp_val = measured_values[:, 0] * 180/np.pi

        difference_sim_minus_exp = sim_val-exp_val

        bins = np.linspace(-tth_bounds, tth_bounds, histogram_nbins)
        fig, ax = plt.subplots()
        plt.hist(difference_sim_minus_exp, bins)
        plt.xlabel(r'$2\theta_{Sim}-2\theta_{Exp}$(degrees)')
        plt.ylabel('Number of Spots')
        plt.xlim([-tth_bounds, tth_bounds])
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$2\theta Error Histogram' + '.png', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.scatter(x, y, c=difference_sim_minus_exp, cmap='RdBu')
        cbar = plt.colorbar()
        cbar.set_label(r'$2\theta_{Sim}-2\theta_{Exp}$(degrees)', rotation=90)
        plt.clim([-tth_bounds, tth_bounds])
        plt.axis('equal')
        plt.xlabel('$X^{D}$ (mm)')
        plt.ylabel('$Y_{D}$ (mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$2\theta Error Spatial Map' + '.png', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.scatter(exp_val, difference_sim_minus_exp)
        plt.ylim([-tth_bounds, tth_bounds])
        plt.xlabel(r'$2\theta_{Exp}$(degrees)')
        plt.ylabel(r'$2\theta_{Sim}-2\theta_{Exp}$(degrees)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$2\theta Error vs Exp' + '.png', dpi=300, bbox_inches='tight')

    elif index_to_plot == 1:  # Eta
        sim_val = predicted_values[:, 1] * 180/np.pi
        exp_val = measured_values[:, 1] * 180/np.pi

        difference_sim_minus_exp = sim_val-exp_val

        bins = np.linspace(-eta_bounds, eta_bounds, histogram_nbins)
        fig, ax = plt.subplots()
        plt.hist(difference_sim_minus_exp, bins)
        plt.xlabel(r'$\eta_{Sim}-\eta_{Exp}$(degrees)')
        plt.ylabel('Number of Spots')
        plt.xlim([-eta_bounds, eta_bounds])
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$\eta Error Histogram' + '.png', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.scatter(x, y, c=difference_sim_minus_exp, cmap='RdBu')
        cbar = plt.colorbar()
        cbar.set_label(r'$\eta_{Sim}-\eta_{Exp}$(degrees)', rotation=90)
        plt.clim([-eta_bounds, eta_bounds])
        plt.axis('equal')
        plt.xlabel('$X^{D}$ (mm)')
        plt.ylabel('$Y_{D}$ (mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$\eta Error Spatial Map' + '.png', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.scatter(exp_val, difference_sim_minus_exp)
        plt.ylim([-eta_bounds, eta_bounds])
        plt.xlabel(r'$\eta_{Exp}$(degrees)')
        plt.ylabel(r'$\eta_{Sim}-\eta_{Exp}$(degrees)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$\eta Error vs Exp' + '.png', dpi=300, bbox_inches='tight')

    elif index_to_plot == 2:  # Omega
        sim_val = predicted_values[:, 2] * 180/np.pi
        exp_val = measured_values[:, 2] * 180/np.pi

        difference_sim_minus_exp = sim_val-exp_val

        bins = np.linspace(-ome_bounds, ome_bounds, histogram_nbins)
        fig, ax = plt.subplots()
        plt.hist(difference_sim_minus_exp, bins)
        plt.xlabel(r'$\omega_{Sim}-\omega_{Exp}$(degrees)')
        plt.ylabel('Number of Spots')
        plt.xlim([-ome_bounds, ome_bounds])
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$\omega Error Histogram' + '.png', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.scatter(x, y, c=difference_sim_minus_exp, cmap='RdBu')
        cbar = plt.colorbar()
        cbar.set_label(r'$\omega_{Sim}-\omega_{Exp}$(degrees)', rotation=90)
        plt.clim([-ome_bounds, ome_bounds])
        plt.axis('equal')
        plt.xlabel('$X^{D}$ (mm)')
        plt.ylabel('$Y_{D}$ (mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$\omega Error Spatial Map' + '.png', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.scatter(exp_val, difference_sim_minus_exp, s=1)
        plt.ylim([-ome_bounds, ome_bounds])
        plt.xlabel(r'$\omega_{Exp}$(degrees)')
        plt.ylabel(r'$\omega_{Sim}-\omega_{Exp}$(degrees)')
        # Write a slope to omega if one exists
        slope, intercept = np.polyfit(exp_val, difference_sim_minus_exp, 1)
        str_to_plot = f'Slope = {slope} deg/deg'
        # Define properties for the text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, str_to_plot, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props)
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$\omega Error vs Exp' + '.png', dpi=300, bbox_inches='tight')

    elif index_to_plot == 3:  # X
        sim_val = predicted_values[:, 3]
        exp_val = measured_values[:, 3]

        difference_sim_minus_exp = sim_val-exp_val

        bins = np.linspace(-translation_bounds,
                           translation_bounds, histogram_nbins)
        fig, ax = plt.subplots()
        plt.hist(difference_sim_minus_exp, bins)
        plt.xlabel(r'$X_{Sim}-X_{Exp}$ (mm)')
        plt.ylabel('Number of Spots')
        plt.xlim([-translation_bounds, translation_bounds])
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$X Error Histogram' + '.png', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.scatter(x, y, c=difference_sim_minus_exp, cmap='RdBu')
        cbar = plt.colorbar()
        cbar.set_label(r'$X_{Sim}-X_{Exp}$(mm)', rotation=90)
        plt.clim([-translation_bounds, translation_bounds])
        plt.axis('equal')
        plt.xlabel('$X^{D}$ (mm)')
        plt.ylabel('$Y_{D}$ (mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$X Error Spatial Map' + '.png', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.scatter(exp_val, difference_sim_minus_exp)
        plt.ylim([-translation_bounds, translation_bounds])
        plt.xlabel(r'$X_{Exp}$(mm)')
        plt.ylabel(r'$X_{Sim}-X_{Exp}$(mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$X Error vs Exp' + '.png', dpi=300, bbox_inches='tight')

    elif index_to_plot == 4:  # Y
        sim_val = predicted_values[:, 4]
        exp_val = measured_values[:, 4]

        difference_sim_minus_exp = sim_val-exp_val

        bins = np.linspace(-translation_bounds,
                           translation_bounds, histogram_nbins)
        fig, ax = plt.subplots()
        plt.hist(difference_sim_minus_exp, bins)
        plt.xlabel(r'$Y_{Sim}-Y_{Exp}$(mm)')
        plt.ylabel('Number of Spots')
        plt.xlim([-translation_bounds, translation_bounds])
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$Y Error Histogram' + '.png', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.scatter(x, y, c=difference_sim_minus_exp, cmap='RdBu')
        cbar = plt.colorbar()
        cbar.set_label(r'$Y_{Sim}-Y_{Exp}$(mm)', rotation=90)
        plt.clim([-translation_bounds, translation_bounds])
        plt.axis('equal')
        plt.xlabel('$X^{D}$ (mm)')
        plt.ylabel('$Y_{D}$ (mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$Y Error Spatial Map' + '.png', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.scatter(exp_val, difference_sim_minus_exp)
        plt.ylim([-translation_bounds, translation_bounds])
        plt.xlabel(r'$Y_{Exp}$(mm)')
        plt.ylabel(r'$Y_{Sim}-Y_{Exp}$(mm)')
        if output_filepath is not None:
            plt.savefig(output_filepath + '/' +
                        r'$Y Error vs Exp' + '.png', dpi=300, bbox_inches='tight')


# %% =====================================
# USER INFO AND SCRIPT EXECUTION - CAN EDIT
# =======================================
# Paths and detector information
instrument_filepath = 'dexelas_V.hexrd'  # 'dexelas_ruby.hexrd'
# grains_out_directory_path = '/usr/workspace/lim37/CHESS_2026-03/mruby-0304-1/mruby-0304-1/'
grains_out_directory_path = '/usr/workspace/lim37/CHESS_2026-03/V-1/V-1a_S0_blob_dog/'
# Choose 'eiger', 'dexela_monoliths', or 'dexela_subpanels'
detector = 'dexela_subpanels'

# Plotting information (bounds are +- the value)
histogram_nbins = 50
translation_bounds = 0.1
tth_bounds = 0.01
eta_bounds = 0.1
ome_bounds = 0.25

# None to not save any images or directory to place you want images dumped (will make a directory if none exists)
# output_filepath = '/usr/workspace/lim37/ER-disloc/voisin-4783-a/SS316L-3/FF/Calibration_figures'


output_filepath = 'V-1/det_cal_check'  # 'Calibration_figures/'
# Script execution
for index_to_plot in range(5):
    generate_cal_check_plots(output_filepath, instrument_filepath, grains_out_directory_path, index_to_plot, detector, histogram_nbins=histogram_nbins,
                             translation_bounds=translation_bounds, tth_bounds=tth_bounds, eta_bounds=eta_bounds, ome_bounds=ome_bounds)


# %%
