"""Shared file handling for the FF stitching and recompile scripts

stitch_grains.py runs once, for the first FF state, and
recompile_grains.py runs for every state after it.  Both write the same files
for the next state (the per-layer seed grains.out files, the lists of which
grains are in each layer and the hexrd fit-grains config yaml), so the file
names and the writers live here.

For a state n the hexrd analysis of layer l is <stem>S<n>_L<l>/grains.out, the
compiled grains are <stem>S<n:03d>_master.out, and the files that seed state
n + 1 are S<n+1>_seed_L<l>_grains.out and grain_nums_S<n>_L<l>.txt.

It also holds the beam calibration that calc_lparm.py makes and the two scripts
use: the correction of the strains for the energy gradient of the beam and the
height of the center of the beam.
"""
import os

import h5py
import numpy as np
import yaml
from scipy.interpolate import BSpline, make_lsq_spline

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


def grains_file_path(ff_dir, analysis_stem, state_num, layer_num):
    """grains.out that hexrd wrote for a layer, in the directory of its analysis in ff_dir"""
    return os.path.join(ff_dir, analysis_name(analysis_stem, state_num, layer_num), 'grains.out')


def master_file_name(analysis_stem, state_num):
    return '%sS%03d_master.out' % (analysis_stem, state_num)


# Where the seed grains of a layer are taken from: the volume that the beam illuminates, which
# is the beam height around the center of the beam, and a margin beyond it (see layer_bounds).
# y is up in the lab frame, so 'above' is toward higher y and 'below' toward lower y.
#   'both'  - the margin is above and below the illuminated volume
#   'above' - the margin is only above it
#   'below' - the margin is only below it
LOCATIONS = ('both', 'above', 'below')


def seed_file_name(state_num, layer_num, location='both'):
    """Seed grains.out of a layer; state_num is the state that it seeds"""
    where = {'both': '', 'above': 'above_', 'below': 'below_'}[location]
    return 'S%d_seed_%sL%d_grains.out' % (state_num, where, layer_num)


def grain_nums_file_name(state_num, layer_num):
    """Which rows of the state's master file are in a layer"""
    return 'grain_nums_S%d_L%d.txt' % (state_num, layer_num)


def yaml_file_name(analysis_stem, state_num):
    return '%sS%d.yml' % (analysis_stem, state_num)


def beam_calibration_file_name(analysis_stem):
    return '%sbeam_calibration.yml' % analysis_stem


# Energy gradient of the beam
#
# The beam energy changes with the height in the beam, so grains are fit with
# the wrong energy by an amount that depends on where in the beam they are.
# That scales all the lattice spacings together, an error in the mean normal
# strain, (ln V[0,0] + ln V[1,1] + ln V[2,2]) / 3, that depends only on y.  The
# gradient is a curve of y fit to that mean strain of the good grains: a
# polynomial of degree 1 for a monochromator with a linear gradient, or a cubic
# spline for a multilayer, whose gradient is not linear and whose ends a
# polynomial would not hold.  y is the grain's tVec_c[1] in the frame of its
# layer, as in a grains.out from hexrd, so it has to be applied before the
# layer offsets.  Outside the range of the grains that were fit the gradient is
# held at its value at the end of the range.

def _fit_curve(y, strain, model, degree, n_knots):
    """Curve of strain against y, as the entries of the gradient dict"""
    if model == 'polynomial':
        # highest power first, as numpy.polyval wants them
        return {'model': 'polynomial', 'degree': int(degree),
                'coefficients': [float(c) for c in np.polyfit(y, strain, degree)]}
    if model == 'spline':
        # least-squares cubic spline with n_knots interior knots, spaced so that each
        # interval has the same number of grains
        order = np.argsort(y)
        x, v = y[order], strain[order]
        interior = np.quantile(x, np.linspace(0, 1, n_knots + 2)[1:-1])
        knots = np.r_[[x[0]] * 4, interior, [x[-1]] * 4]
        spline = make_lsq_spline(x, v, knots, k=3)
        return {'model': 'spline', 'n_knots': int(n_knots), 'knots': [float(t) for t in spline.t],
                'coefficients': [float(c) for c in spline.c]}
    raise ValueError("model has to be 'spline' or 'polynomial', not %r" % (model,))


def _eval_curve(curve, y):
    if curve['model'] == 'polynomial':
        return np.polyval(curve['coefficients'], y)
    return BSpline(np.array(curve['knots']), np.array(curve['coefficients']), 3)(y)


def well_populated_range(y, bin_width, min_grains):
    """(lower, upper) of y, around the busiest bin, within which every bin has min_grains

    Beyond this there are too few grains for the gradient to be fit to, and the ends
    of a fit that reached out to them would rest on a few grains.
    """
    edges = np.arange(y.min(), y.max() + bin_width, bin_width)
    counts, edges = np.histogram(y, edges)
    lo = hi = int(np.argmax(counts))
    while lo > 0 and counts[lo - 1] >= min_grains:
        lo -= 1
    while hi < len(counts) - 1 and counts[hi + 1] >= min_grains:
        hi += 1
    return edges[lo], edges[hi + 1]


def fit_energy_gradient(y, mean_strain, model='spline', degree=1, n_knots=6, outlier_sigma=None):
    """Curve of the mean normal strain against y

    model is 'spline' (n_knots interior knots) or 'polynomial' (of the given
    degree).  Grains that are more than outlier_sigma standard deviations from
    the fit are dropped, and the fit is repeated, until none are.  Returns a
    dict that save_beam_calibration writes.
    """
    y = np.asarray(y, float)
    mean_strain = np.asarray(mean_strain, float)
    use = np.ones(len(y), bool)
    for _ in range(20):
        curve = _fit_curve(y[use], mean_strain[use], model, degree, n_knots)
        resid = mean_strain - _eval_curve(curve, np.clip(y, y[use].min(), y[use].max()))
        rms = resid[use].std()
        new_use = np.abs(resid) <= outlier_sigma * rms if outlier_sigma else use
        if np.array_equal(new_use, use):
            break
        use = new_use
    curve.update({
        # the fit is only good where there are grains, so y is held to this range
        'y_min': float(y[use].min()),
        'y_max': float(y[use].max()),
        'n_grains': int(use.sum()),
        'rms_residual': float(rms),
        'mean_strain': float(mean_strain[use].mean()),
        'used': use,
    })
    return curve


def save_beam_calibration(file_name, calibration):
    """Write the energy gradient from fit_energy_gradient, with the 'beam_center_y' added to it"""
    # 'used' marks the grains of the fit, which only the diagnostic plot needs
    saved = {k: v for k, v in calibration.items() if k != 'used'}
    with open(file_name, 'w') as f:
        yaml.dump(saved, f, default_flow_style=False)


def load_beam_calibration(file_name):
    """The energy gradient and the 'beam_center_y', the height of the center of the beam in
    the frame of a layer, from the file calc_lparm.py wrote"""
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            'No beam calibration file %s, run calc_lparm.py first or set beam_calibration_file to None '
            'for no correction' % file_name)
    with open(file_name, 'r') as f:
        return yaml.safe_load(f)


def energy_gradient_strain(gradient, y):
    """Mean normal strain error of the energy gradient at y, held to the fit range"""
    return _eval_curve(gradient, np.clip(y, gradient['y_min'], gradient['y_max']))


def correct_energy_gradient(grain_data, gradient):
    """Copy of grain_data with the energy gradient taken out of the strains

    The strain at the grain's y is removed from each of the three normal
    components, ln V[0,0], ln V[1,1] and ln V[2,2] (columns 15 to 17).
    """
    corrected = np.array(grain_data, dtype=float)
    corrected[:, 15:18] -= energy_gradient_strain(gradient, corrected[:, 7])[:, None]
    return corrected


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


def read_lattice_parameter(materials_file, material_name):
    """Lattice parameter a of a material in a hexrd materials file, in angstrom"""
    with h5py.File(materials_file, 'r') as f:
        return 10.0 * float(f[material_name]['LatticeParameters'][0])  # the file has nm


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


def layer_bounds(vert_off, beam_height, location='both', v_pos_corr=0.0, beam_center_y=0.0,
                 seed_margin=0.5):
    """Y range, (lower, upper), of the grains that are taken as seeds for each layer

    This is the volume that the beam illuminates in the layer, beam_height around the
    center of the beam (beam_center_y above the middle of the layer, in the frame of the
    layer), and a margin of seed_margin beam heights beyond it, on the side(s) that
    location asks for, see LOCATIONS.  The margin takes in grains from just outside of
    the illuminated volume, which can move into it.  The highest layer has no upper
    limit and the lowest layer has no lower limit, so that the grains beyond the layers
    on the outside are included too.
    """
    if location not in LOCATIONS:
        raise ValueError('location has to be one of %s, not %r' % (', '.join(LOCATIONS), location))
    margin = seed_margin * beam_height
    margin_below = margin if location in ('both', 'below') else 0.0
    margin_above = margin if location in ('both', 'above') else 0.0
    bounds = []
    for off in vert_off:
        center = off + beam_center_y - v_pos_corr
        bounds.append((center - beam_height / 2 - margin_below,
                       center + beam_height / 2 + margin_above))
    bounds[int(np.argmax(vert_off))] = (bounds[int(np.argmax(vert_off))][0], np.inf)
    bounds[int(np.argmin(vert_off))] = (-np.inf, bounds[int(np.argmin(vert_off))][1])
    return bounds


def make_grain_lists(analysis_stem, grain_data, state_num, beam_height, pos_offset, output_dir,
                     location='both', v_pos_corr=0.0, low_comp_thresh=None, high_chi2_thresh=None,
                     beam_center_y=0.0, seed_margin=0.5):
    """Make the seed grains.out file and grain number list of each layer for state_num + 1

    grain_data are the compiled grains of state_num, with positions in the
    sample frame.  A grain is in a layer if it is within the y bounds of
    layer_bounds, which are the illuminated volume and the margin of seed_margin
    beam heights on the side(s) that location asks for, and, when the thresholds
    are given, has completeness of at least low_comp_thresh and chi2 below
    high_chi2_thresh.  The seed grains are moved into the frame of their layer,
    which keeps the beam center at beam_center_y.
    """
    vert_off = pos_offset[1]
    y_loc = grain_data[:, 7]
    keep = np.ones(len(grain_data), bool)
    if low_comp_thresh is not None:
        keep &= grain_data[:, 1] >= low_comp_thresh
    if high_chi2_thresh is not None:
        keep &= grain_data[:, 2] < high_chi2_thresh
    print('There are %d grains in the compiled list' % len(grain_data))

    bounds = layer_bounds(vert_off, beam_height, location, v_pos_corr, beam_center_y, seed_margin)
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
