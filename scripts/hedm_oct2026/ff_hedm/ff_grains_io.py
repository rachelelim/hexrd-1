"""Shared file handling for the FF stitching and recompile scripts

stitch_grains_clean_Apr25.py runs once, for the first FF state, and
recompile_grains.py runs for every state after it.  Both write the same files
for the next state (the per-layer seed grains.out files, the lists of which
grains are in each layer and the hexrd fit-grains config yaml), so the file
names and the writers live here.

For a state n the hexrd analysis of layer l is <stem>S<n>_L<l>/grains.out, the
compiled grains are <stem>S<n:03d>_master.out, and the files that seed state
n + 1 are S<n+1>_seed_L<l>_grains.out and grain_nums_S<n>_L<l>.txt.
"""
import os

import h5py
import numpy as np
import yaml

GRAINS_HEADER = (
    "grain ID", "completeness", "chi2", "xi[0]", "xi[1]", "xi[2]",
    "tVec_c[0]", "tVec_c[1]", "tVec_c[2]",
    "vInv_s[0,0]", "vInv_s[1,1]", "vInv_s[2,2]",
    "vInv_s[1,2]*sqrt(2)", "vInv_s[0,2]*sqrt(2)", "vInv_s[0,1]*sqrt(2)",
    "ln(V[0,0])", "ln(V[1,1])", "ln(V[2,2])",
    "ln(V[1,2])", "ln(V[0,2])", "ln(V[0,1])",
)


# File names

def analysis_name(analysis_stem, state_num, layer_num):
    """Name of the hexrd analysis (and its directory) for one layer of a state"""
    return '%sS%d_L%d' % (analysis_stem, state_num, layer_num)


def master_file_name(analysis_stem, state_num):
    return '%sS%03d_master.out' % (analysis_stem, state_num)


def seed_file_name(state_num, layer_num, location='both'):
    """Seed grains.out of a layer; state_num is the state that it seeds"""
    where = {'both': '', 'top': 'top_', 'bottom': 'bottom_'}[location]
    return 'S%d_seed_%sL%d_grains.out' % (state_num, where, layer_num)


def grain_nums_file_name(state_num, layer_num):
    """Which rows of the state's master file are in a layer"""
    return 'grain_nums_S%d_L%d.txt' % (state_num, layer_num)


def yaml_file_name(analysis_stem, state_num):
    return '%sS%d.yml' % (analysis_stem, state_num)


# Instrument file

def read_detector_panels(instrument_file):
    """Panels of each detector in a hexrd instrument file, {detector key: [panel names]}

    The detector key is the panel's 'group' (ff1 for ff1_0_0, ff1_0_1, ...),
    or the panel name itself for panels that are not grouped.
    """
    det_panels = {}
    with h5py.File(instrument_file, 'r') as f:
        for panel in f['instrument/detectors']:
            grp = f['instrument/detectors'][panel]
            key = grp['group'][()].decode() if 'group' in grp else panel
            det_panels.setdefault(key, []).append(panel)
    return det_panels


def read_beam_energy(instrument_file):
    """Beam energy in keV from a hexrd instrument file"""
    with h5py.File(instrument_file, 'r') as f:
        return float(np.squeeze(f['instrument/beam/energy'][()]))


# Writers

def write_grains_files(output_dir, analysis_stem, grain_data, state_num, layer_num=None, location='both'):
    """Write a grains.out style file

    Without layer_num this is the master file of state_num, with it the seed
    file of that layer for state_num + 1.
    """
    if layer_num is None:
        file_name = master_file_name(analysis_stem, state_num)
    else:
        ngrains = len(grain_data)
        print('\t %d grains in layer %d' % (ngrains, layer_num))
        grain_data[:, 0] = np.arange(ngrains)
        file_name = seed_file_name(state_num + 1, layer_num, location)

    len_items = [max(len(item), 19) for item in GRAINS_HEADER[1:]]  # for %19.12g
    header_fmt = "#%13s  " + "  ".join(["%%%ds" % n for n in len_items]) + "\n"
    row_fmt = "%14d  " + "  ".join(["%%%d.12g" % n for n in len_items]) + "\n"
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write(header_fmt % GRAINS_HEADER)
        for row in grain_data[:, :21]:
            f.write(row_fmt % tuple(row))


def layer_bounds(vert_off, beam_height, location='both', v_pos_corr=0.0):
    """Y range, (lower, upper), of the grains that can be in each layer

    The range reaches 1.5 beam heights to the side(s) of the layer that
    'location' asks for (below for 'bottom', above for 'top', both for 'both').
    The first layer has no upper limit.
    """
    below, above = {'bottom': (1.5, 0.5), 'top': (0.5, 1.5), 'both': (1.5, 1.5)}[location]
    bounds = []
    for i, off in enumerate(vert_off):
        lower = off - below*beam_height - v_pos_corr
        upper = np.inf if i == 0 else off + above*beam_height - v_pos_corr
        bounds.append((lower, upper))
    return bounds


def make_grain_lists(analysis_stem, grain_data, state_num, beam_height, pos_offset, output_dir,
                     location='both', v_pos_corr=0.0, low_comp_thresh=None, high_chi2_thresh=None):
    """Make the seed grains.out file and grain number list of each layer for state_num + 1

    grain_data are the compiled grains of state_num, with positions in the
    sample frame.  A grain is in a layer if it is within the y bounds of
    layer_bounds, and, when the thresholds are given, has completeness of at
    least low_comp_thresh and chi2 below high_chi2_thresh.  The seed grains are
    moved into the frame of their layer.
    """
    vert_off = pos_offset[1]
    y_loc = grain_data[:, 7]
    keep = np.ones(len(grain_data), bool)
    if low_comp_thresh is not None:
        keep &= grain_data[:, 1] >= low_comp_thresh
    if high_chi2_thresh is not None:
        keep &= grain_data[:, 2] < high_chi2_thresh
    print('There are %d grains in the compiled list' % len(grain_data))

    bounds = layer_bounds(vert_off, beam_height, location, v_pos_corr)
    for i, (lower, upper) in enumerate(bounds):
        used = np.where(keep & (y_loc >= lower) & (y_loc <= upper))[0]
        layer_grain_data = grain_data[used, :]
        layer_grain_data[:, 7] = layer_grain_data[:, 7] - vert_off[i]

        np.savetxt(os.path.join(output_dir, grain_nums_file_name(state_num, i)), used.astype(int))
        write_grains_files(output_dir, analysis_stem, layer_grain_data,
                           state_num, layer_num=i, location=location)


def write_yaml(init_yaml, image_fstem, sample, analysis_stem, state_num, num_layers, first_scan_num,
               first_file_num, output_dir, det_panels, location='both'):
    """Write the hexrd config of state_num, one analysis per layer

    The first layer is the full config from init_yaml and each later layer
    only updates the name, image files and fit-grains estimate.  image_fstem is
    filled with (sample, scan number, detector key, file number) and det_panels
    is {detector key: [panel names]}, see read_detector_panels.
    """
    with open(init_yaml, "r") as f:
        initial_yaml_info = yaml.load(f, Loader=yaml.Loader)
    yaml_dict = [initial_yaml_info]

    for layer_num in range(num_layers):
        image_list = [{'file': image_fstem % (sample, first_scan_num + layer_num, det, first_file_num + layer_num),
                       'args': {},
                       'panel': panels}
                      for det, panels in det_panels.items()]
        name = analysis_name(analysis_stem, state_num, layer_num)
        estimate = seed_file_name(state_num, layer_num, location)

        if layer_num == 0:
            initial_yaml_info["analysis_name"] = name
            initial_yaml_info['image_series']["data"] = image_list
            initial_yaml_info["fit_grains"]["estimate"] = estimate
        else:
            yaml_dict.append({"analysis_name": name,
                              "image_series": {"data": image_list},
                              "fit_grains": {"estimate": estimate}})

    with open(os.path.join(output_dir, yaml_file_name(analysis_stem, state_num)), "w") as f:
        yaml.dump_all(yaml_dict, f, default_flow_style=False)
