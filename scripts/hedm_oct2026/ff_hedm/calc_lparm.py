"""Fit the energy gradient of the beam and find its center, from the grains of the first state

The beam energy changes with the height in the beam, which puts a strain error on every grain that
depends on where in the beam it is.  The gradient is linear for some monochromators and not for others,
such as the multilayers, so it is fit as a line (a polynomial of degree 1) for a linear one and as a
spline for a multilayer.

This fits the mean normal strain of the good grains against their y in the beam, with all of the layers
together, and saves the curve for stitch_grains.py and recompile_grains.py to take out of the strains,
with a diagnostic figure of the fit.  It also finds the height of the center of the beam in the frame of
a layer, the middle of the heights where there are enough grains, which they use to place the windows of
the seed grains of each layer.  It assumes that the sample has the same mean strain at every height, so
that the trend with y is the gradient and not the sample.

Run it once, on the first state, before stitch_grains.py:

    python calc_lparm.py ff_pipeline.yml [--state 0] [--show]

Author: lim37
"""

import argparse
import os

import numpy as np

from ff_config import load_config
from ff_grains_io import (
    beam_calibration_file_name, energy_gradient_strain, fit_energy_gradient,
    grains_file_path, read_lattice_parameter, save_beam_calibration, well_populated_range,
)


def calibrate(cfg, state=0, show=False):
    """Fit the beam calibration from the grains of a state and save it, with the figures

    cfg is from ff_config.load_config.  Returns the calibration, which is what is saved.
    """
    c = cfg.calibration
    y_all, strain_all, layer_all, every_grain = [], [], [], []
    for i in range(cfg.num_layers):
        data = np.loadtxt(grains_file_path(cfg.ff_dir, cfg.analysis_stem, state, i))
        good = np.logical_and(data[:, 2] < c.good_chi2_max, data[:, 1] > c.good_comp_min)
        print('Layer %d: %d of %d grains are good' % (i, good.sum(), len(data)))
        # y, completeness, whether it is good and the layer, of every grain, for the density figure
        every_grain.append(np.column_stack([data[:, 7], data[:, 1], good, np.full(len(data), i)]))
        y_all.append(data[good, 7])
        strain_all.append(np.mean(data[good, 15:18], axis=1))
        layer_all.append(np.full(good.sum(), i))
    y_all = np.concatenate(y_all)
    strain_all = np.concatenate(strain_all)
    layer_all = np.concatenate(layer_all)
    every_grain = np.concatenate(every_grain)

    def fit(y, strain, model, size):
        """size is the number of knots of a spline or the degree of a polynomial"""
        return fit_energy_gradient(y, strain, model=model, degree=size, n_knots=size,
                                   outlier_sigma=c.outlier_sigma)

    y_lo, y_hi = well_populated_range(y_all, c.trim_bin_width, c.min_grains_per_bin)
    in_range = np.logical_and(y_all >= y_lo, y_all < y_hi)
    print('%d of %d good grains are in y from %.3f to %.3f, where every %g mm bin has %d grains'
          % (in_range.sum(), len(y_all), y_lo, y_hi, c.trim_bin_width, c.min_grains_per_bin))
    # the grains that are not in the range, or that are outliers, are not used
    y_fit, strain_fit = y_all[in_range], strain_all[in_range]

    gradient = fit(y_fit, strain_fit, c.model, c.n_knots if c.model == 'spline' else c.poly_degree)
    used = np.zeros(len(y_all), bool)
    used[in_range] = gradient['used']
    print('%s fit to %d of %d grains, rms %.2e, y from %.3f to %.3f'
          % (c.model, gradient['n_grains'], len(y_all),
             gradient['rms_residual'], gradient['y_min'], gradient['y_max']))
    L0 = read_lattice_parameter(cfg.materials_file, cfg.material_name)
    print('Mean normal strain %.4e, lattice parameter %.5f (%.5f in the materials file)' %
          (gradient['mean_strain'], L0 * (1 + gradient['mean_strain']), L0))

    # The beam is as high as the range of heights with enough grains, and its center is the middle of it
    beam_center_y = 0.5 * (y_lo + y_hi)
    gradient['beam_center_y'] = float(beam_center_y)
    print('Beam center at y = %.3f mm, %.3f mm high' % (beam_center_y, y_hi - y_lo))

    os.makedirs(cfg.output_dir, exist_ok=True)
    calibration_file = os.path.join(cfg.output_dir, beam_calibration_file_name(cfg.analysis_stem))
    save_beam_calibration(calibration_file, gradient)
    print('Saved %s' % calibration_file)

    # Cross-validation, to compare the models
    def cross_validated_rms(model, size, folds=5):
        """rms of the grains that were left out of each fit, which are scored where
        the fit holds, so grains beyond the ends are held to the ends"""
        order = np.random.default_rng(0).permutation(len(y_fit))
        err = []
        for k in range(folds):
            test = np.zeros(len(y_fit), bool)
            test[order[k::folds]] = True
            g = fit(y_fit[~test], strain_fit[~test], model, size)
            err.append(strain_fit[test] - energy_gradient_strain(g, y_fit[test]))
        return np.sqrt(np.mean(np.concatenate(err) ** 2))

    knot_counts = np.arange(0, 13)
    degrees = np.arange(1, 9)
    cv_spline = [cross_validated_rms('spline', n) for n in knot_counts]
    cv_poly = [cross_validated_rms('polynomial', d) for d in degrees]
    # the fewest knots that do about as well as the best
    enough = knot_counts[np.argmax(np.array(cv_spline) <= 1.01 * np.min(cv_spline))]
    print('Cross-validated rms of the spline is within 1%% of its best (%.2e) from %d interior knots'
          % (np.min(cv_spline), enough))

    if not show:
        import matplotlib
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    colors = ['C%d' % i for i in range(cfg.num_layers)]
    num_layers = cfg.num_layers
    samp, state_num = cfg.sample, state

    # Diagnostic figure of the fit
    ys = np.linspace(y_all.min(), y_all.max(), 400)
    fit_ys = energy_gradient_strain(gradient, ys)
    resid = strain_all - energy_gradient_strain(gradient, y_all)
    colors = ['C%d' % i for i in range(num_layers)]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    a = ax[0, 0]
    for i in range(num_layers):
        m = np.logical_and(layer_all == i, used)
        a.plot(y_all[m], strain_all[m], '.', color=colors[i], label='layer %d' % i)
    a.plot(y_all[~used], strain_all[~used], 'kx', label='left out of the fit (too few grains or outliers)')
    a.plot(ys, fit_ys, 'k-', lw=2, label='gradient (%s)' % gradient['model'])
    a.axvspan(ys[0], gradient['y_min'], color='0.9')
    a.axvspan(gradient['y_max'], ys[-1], color='0.9')
    a.axvline(beam_center_y, color='C4', ls='--', label='beam center')
    a.set_xlabel('y in the beam (mm)')
    a.set_ylabel('mean of ln V[0,0], ln V[1,1], ln V[2,2]')
    a.set_title('Strain before the correction (grey: held at the end value)')
    a.legend(fontsize=8, loc='lower right')

    a = ax[0, 1]
    for i in range(num_layers):
        m = np.logical_and(layer_all == i, used)
        a.plot(y_all[m], resid[m], '.', color=colors[i])
    a.plot(y_all[~used], resid[~used], 'kx')
    a.axhline(0, color='k')
    a.axvline(beam_center_y, color='C4', ls='--')
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
    figure_file = os.path.join(cfg.output_dir, '%senergy_gradient.png' % cfg.analysis_stem)
    fig.savefig(figure_file, dpi=150)
    print('Saved %s' % figure_file)

    # Density of grains in the beam.  Only a diagnostic, nothing in the analysis uses it.  How many
    # grains are found at each height in the beam follows the intensity there, which is not uniform,
    # and the grains where it is weak are also the ones that are harder to fit.
    yg, comp_g, good_g, layer_g = every_grain.T
    good_g = good_g.astype(bool)
    edges = np.arange(yg.min(), yg.max() + c.density_bin_width, c.density_bin_width)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for a in ax:
        a.axvline(beam_center_y, color='C4', ls='--')

    a = ax[0]
    for i in range(num_layers):
        counts, _ = np.histogram(yg[layer_g == i], edges)
        a.step(centers, counts / c.density_bin_width, where='mid', color=colors[i], label='layer %d' % i)
    counts, _ = np.histogram(yg, edges)
    a.step(centers, counts / c.density_bin_width, where='mid', color='k', lw=2, label='all layers')
    a.set_ylabel('grains per mm')
    a.set_title('Grains in the files (dashed: beam center %.3f mm)' % beam_center_y)
    a.legend(fontsize=8)

    a = ax[1]
    counts_all, _ = np.histogram(yg, edges)
    counts_good, _ = np.histogram(yg[good_g], edges)
    a.step(centers, counts_all / c.density_bin_width, where='mid', color='0.6', label='all grains in the files')
    a.step(centers, counts_good / c.density_bin_width, where='mid', color='k',
           label='good grains (completeness > %g, chi2 < %g)' % (c.good_comp_min, c.good_chi2_max))
    a.set_ylabel('grains per mm, all layers')
    a.set_title('Good grains')
    a.legend(fontsize=8, loc='upper right')

    a = ax[2]
    which = np.digitize(yg, edges) - 1
    mean_comp = np.array([comp_g[which == k].mean() if np.any(which == k) else np.nan
                          for k in range(len(centers))])
    line_comp, = a.plot(centers, mean_comp, 'o-', ms=3, color='C0', label='mean completeness')
    a.set_ylabel('mean completeness')
    a.set_xlabel('y in the beam (mm)')
    b = a.twinx()
    line_good, = b.plot(centers, counts_good / np.maximum(counts_all, 1), 's-', ms=3, color='C1',
                        label='fraction good')
    b.set_ylabel('fraction of grains that are good')
    a.set_title('How well the grains fit')
    a.legend(handles=[line_comp, line_good], loc='lower center', fontsize=8)


    fig.suptitle('Grains in the beam of %s, state %d' % (samp, state_num))
    fig.tight_layout()
    density_file = os.path.join(cfg.output_dir, '%sgrain_density.png' % cfg.analysis_stem)
    fig.savefig(density_file, dpi=150)
    print('Saved %s' % density_file)
    if show:
        plt.show()
    plt.close('all')
    return gradient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('config', help='the pipeline config yaml')
    parser.add_argument('--state', type=int, default=0, help='the state to fit from, 0 for the first')
    parser.add_argument('--show', action='store_true', help='also show the figures')
    args = parser.parse_args(argv)
    calibrate(load_config(args.config), args.state, args.show)


if __name__ == '__main__':
    main()
