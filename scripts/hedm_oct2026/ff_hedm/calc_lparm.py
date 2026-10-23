"""Fit the energy gradient of the beam from the grains of the first state

The beam energy changes with the height in the beam, which puts a strain error
on every grain that depends on where in the beam it is.  The gradient is
linear for some monochromators and not for others, such as the multilayers, so
it is fit as a line (a polynomial of degree 1) for a linear one and as a spline
for a multilayer.

This fits the mean normal strain of the good grains against their y in the beam,
with all of the layers together, and saves the curve for stitch_grains.py and
recompile_grains.py to take out of the strains, with a diagnostic figure of the
fit.  Run it once, on the first state, before stitch_grains.py.  It assumes that
the sample has the same mean strain at every height, so that the trend with y is
the gradient and not the sample.
"""

# %% Import Modules
import os

import numpy as np
from matplotlib import pyplot as plt

from ff_grains_io import (
    energy_gradient_file_name, energy_gradient_strain, fit_energy_gradient,
    save_energy_gradient,
)


# %% User Input
###############################################################################

samp = 'diamond-1a'
analysis_stem = "%s_" % samp

data_dir = "/Users/lim37/Library/CloudStorage/OneDrive-LLNL/TriMet - General/CHESS_2026-09"
# the grains.out of each layer of the first state, as read by stitch_grains.py
grain_file_stem = data_dir + '/%s/FF/%s_S%%d_L%%d_grains.out' % (samp, samp)
state_num = 0
num_layers = 4

# Grains used for the fit
good_comp_min = 0.85
good_chi2_max = 0.01

# 'polynomial' with poly_degree 1 for a monochromator with a linear gradient, 'spline'
# for a multilayer.  n_knots is the number of interior knots of the spline: more follow
# the gradient more closely, fewer are smoother.  The figure shows how the cross-validated
# error of each changes, to choose it by.
gradient_model = 'spline'
n_knots = 4
poly_degree = 1
# grains this many standard deviations from the fit are left out of it
outlier_sigma = 4

# Lattice parameter of the unstrained material (angstrom), to report the mean one
L0 = 3.5916

output_dir = './'


# %% Fit
###############################################################################

y_all = []
strain_all = []
layer_all = []
for i in range(num_layers):
    data = np.loadtxt(grain_file_stem % (state_num, i))
    good = np.logical_and(data[:, 2] < good_chi2_max, data[:, 1] > good_comp_min)
    print('Layer %d: %d of %d grains are good' % (i, good.sum(), len(data)))
    y_all.append(data[good, 7])
    strain_all.append(np.mean(data[good, 15:18], axis=1))
    layer_all.append(np.full(good.sum(), i))

y_all = np.concatenate(y_all)
strain_all = np.concatenate(strain_all)
layer_all = np.concatenate(layer_all)


def fit(y, strain, model, size):
    """size is the number of knots of a spline or the degree of a polynomial"""
    return fit_energy_gradient(y, strain, model=model, degree=size, n_knots=size,
                               outlier_sigma=outlier_sigma)


gradient = fit(y_all, strain_all, gradient_model,
               n_knots if gradient_model == 'spline' else poly_degree)
used = gradient['used']
print('%s fit to %d of %d grains, rms %.2e, y from %.3f to %.3f'
      % (gradient_model, gradient['n_grains'], len(y_all),
         gradient['rms_residual'], gradient['y_min'], gradient['y_max']))
print('Mean normal strain %.4e, lattice parameter %.5f' %
      (gradient['mean_strain'], L0 * (1 + gradient['mean_strain'])))

os.makedirs(output_dir, exist_ok=True)
gradient_file = os.path.join(output_dir, energy_gradient_file_name(analysis_stem))
save_energy_gradient(gradient_file, gradient)
print('Saved %s' % gradient_file)


# %% Cross-validation, to compare the models
###############################################################################

def cross_validated_rms(model, size, folds=5):
    """rms of the grains that were left out of each fit, which are scored where
    the fit holds, so grains beyond the ends are held to the ends"""
    order = np.random.default_rng(0).permutation(len(y_all))
    err = []
    for k in range(folds):
        test = np.zeros(len(y_all), bool)
        test[order[k::folds]] = True
        g = fit(y_all[~test], strain_all[~test], model, size)
        err.append(strain_all[test] - energy_gradient_strain(g, y_all[test]))
    return np.sqrt(np.mean(np.concatenate(err) ** 2))


knot_counts = np.arange(0, 13)
degrees = np.arange(1, 9)
cv_spline = [cross_validated_rms('spline', n) for n in knot_counts]
cv_poly = [cross_validated_rms('polynomial', d) for d in degrees]
# the fewest knots that do about as well as the best
enough = knot_counts[np.argmax(np.array(cv_spline) <= 1.01 * np.min(cv_spline))]
print('Cross-validated rms of the spline is within 1%% of its best (%.2e) from %d interior knots'
      % (np.min(cv_spline), enough))


# %% Diagnostic figure
###############################################################################

ys = np.linspace(y_all.min(), y_all.max(), 400)
fit_ys = energy_gradient_strain(gradient, ys)
resid = strain_all - energy_gradient_strain(gradient, y_all)
colors = ['C%d' % i for i in range(num_layers)]

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

a = ax[0, 0]
for i in range(num_layers):
    m = np.logical_and(layer_all == i, used)
    a.plot(y_all[m], strain_all[m], '.', color=colors[i], label='layer %d' % i)
a.plot(y_all[~used], strain_all[~used], 'kx', label='left out of the fit')
a.plot(ys, fit_ys, 'k-', lw=2, label='gradient (%s)' % gradient_model)
a.axvspan(ys[0], gradient['y_min'], color='0.9')
a.axvspan(gradient['y_max'], ys[-1], color='0.9')
a.set_xlabel('y in the beam (mm)')
a.set_ylabel('mean of ln V[0,0], ln V[1,1], ln V[2,2]')
a.set_title('Strain before the correction (grey: held at the end value)')
a.legend(fontsize=8)

a = ax[0, 1]
for i in range(num_layers):
    m = np.logical_and(layer_all == i, used)
    a.plot(y_all[m], resid[m], '.', color=colors[i])
a.plot(y_all[~used], resid[~used], 'kx')
a.axhline(0, color='k')
a.axhline(gradient['rms_residual'], color='k', ls=':')
a.axhline(-gradient['rms_residual'], color='k', ls=':')
a.set_xlabel('y in the beam (mm)')
a.set_title('After the correction (dotted: rms %.1e)' % gradient['rms_residual'])

a = ax[1, 0]
bins = np.linspace(y_all.min(), y_all.max(), 25)
for i in range(num_layers):
    m = layer_all == i
    a.hist(y_all[m], bins=bins, histtype='step', color=colors[i],
           label='layer %d, residual %.1e $\\pm$ %.1e' % (
               i, resid[m & used].mean(), resid[m & used].std() / np.sqrt((m & used).sum())))
a.set_xlabel('y in the beam (mm)')
a.set_ylabel('grains')
a.set_title('Where the grains are, and the mean residual of each layer')
a.legend(fontsize=8)

a = ax[1, 1]
a.plot(knot_counts, cv_spline, 'o-', label='spline, interior knots')
a.plot(degrees, cv_poly, 's-', label='polynomial, degree')
a.set_yscale('log')
a.set_xlabel('knots or degree')
a.set_ylabel('cross-validated rms strain')
a.set_title('Model choice')
a.legend(fontsize=8)

fig.suptitle('Energy gradient of %s, state %d' % (samp, state_num))
fig.tight_layout()
figure_file = os.path.join(output_dir, '%senergy_gradient.png' % analysis_stem)
fig.savefig(figure_file, dpi=150)
print('Saved %s' % figure_file)
plt.show()
