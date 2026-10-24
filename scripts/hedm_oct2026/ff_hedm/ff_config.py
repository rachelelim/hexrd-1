"""The settings of the FF pipeline, in one yaml file that every script reads

Every script of the pipeline (calc_lparm.py, stitch_grains.py, recompile_grains.py and
build_pipeline.py) takes the path to this file, so the settings that they have to agree on are
only written once.  See ff_pipeline.yml for an example with every setting, and README.md for
what they do.

Relative paths in the file are relative to the directory that the file is in, so that the scripts can
be run from anywhere.  The exception is hexrd: image_fstem is a name inside the hexrd config, so it
is passed on as it is, and hexrd runs in output_dir.
"""
import os
from types import SimpleNamespace

import numpy as np
import yaml

from ff_grains_io import LOCATIONS, beam_calibration_file_name

# The settings that can be left out, with what they are then.  A setting that is not here or in
# REQUIRED is not allowed, so that a misspelt one is an error and not silently ignored.
DEFAULTS = {
    'seeds': {
        'location': 'both',
        'margin': 0.5,
    },
    'stitch': {
        'low_comp_thresh': 0.5,
        'high_chi2_thresh': 0.05,
        'dist': 0.1,
        'misorientation': 0.25,
        'completeness_diff': 0.1,
        'v_pos_corr': 0.0,
        'make_layer_grain_lists': True,
    },
    'recompile': {
        'close_completeness_cutoff': 0.1,
        'high_chi2_thresh': 0.05,
        'low_comp_thresh': 0.6,
    },
    'calibration': {
        'apply': True,
        'good_comp_min': 0.85,
        'good_chi2_max': 0.01,
        'model': 'spline',
        'n_knots': 4,
        'poly_degree': 1,
        'outlier_sigma': 4,
        'min_grains_per_bin': 10,
        'trim_bin_width': 0.025,
        'density_bin_width': 0.01,
    },
    'hexrd': {
        'command': 'hexrd fit-grains',
        'args': '',
        'initial_yaml': None,
        'image_fstem': None,
    },
}
REQUIRED = {
    'layers': ['ramsz', 'beam_height'],
    'materials': ['file', 'name'],
}
TOP_LEVEL = ['sample', 'analysis_stem', 'ff_dir', 'output_dir', 'detector_file', 'states']


class ConfigError(ValueError):
    pass


def _check_keys(section, given, allowed):
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ConfigError('Unknown setting%s in %s: %s.  The ones there are: %s'
                          % ('s' if len(unknown) > 1 else '', section, ', '.join(unknown),
                             ', '.join(sorted(allowed))))


def _section(raw, name):
    """The settings of a section, with the defaults filled in"""
    given = raw.get(name) or {}
    if not isinstance(given, dict):
        raise ConfigError('%s has to be a list of settings' % name)
    _check_keys(name, given, list(DEFAULTS.get(name, {})) + REQUIRED.get(name, []))
    for key in REQUIRED.get(name, []):
        if key not in given:
            raise ConfigError('%s needs %s' % (name, key))
    return {**DEFAULTS.get(name, {}), **given}


def _path(base_dir, value):
    return os.path.normpath(os.path.join(base_dir, os.path.expanduser(str(value))))


def load_config(file_name):
    """Read and check a pipeline config, returns its settings with the paths worked out

    The settings are attributes: sample, analysis_stem, ff_dir, output_dir, detector_file,
    materials_file, material_name, ramsz, num_layers, pos_offset (3 x num_layers, the y of a layer is
    -ramsz), beam_height, location, seed_margin, beam_calibration_file (None for no calibration),
    states ({state number: (first scan, first file)}), and the sections stitch, recompile, calibration
    and hexrd with their own settings.
    """
    file_name = os.path.abspath(file_name)
    base_dir = os.path.dirname(file_name)
    with open(file_name, 'r') as f:
        raw = yaml.safe_load(f) or {}
    _check_keys('the top level', raw, TOP_LEVEL + list(DEFAULTS) + list(REQUIRED))
    for key in ('sample', 'ff_dir', 'detector_file'):
        if key not in raw:
            raise ConfigError('The config needs %s' % key)

    layers = _section(raw, 'layers')
    materials = _section(raw, 'materials')
    seeds = _section(raw, 'seeds')
    hexrd = _section(raw, 'hexrd')
    calibration = _section(raw, 'calibration')

    ramsz = np.atleast_1d(np.array(layers['ramsz'], dtype=float))
    if ramsz.ndim != 1 or len(ramsz) < 1:
        raise ConfigError('layers: ramsz has to be a list with the ramsz of each layer')
    if seeds['location'] not in LOCATIONS:
        raise ConfigError('seeds: location has to be one of %s, not %r'
                          % (', '.join(LOCATIONS), seeds['location']))
    if seeds['margin'] < 0:
        raise ConfigError('seeds: margin cannot be negative')
    if calibration['model'] not in ('spline', 'polynomial'):
        raise ConfigError("calibration: model has to be 'spline' or 'polynomial'")

    output_dir = _path(base_dir, raw.get('output_dir', '.'))
    analysis_stem = raw.get('analysis_stem', '%s_' % raw['sample'])
    pos_offset = np.zeros([3, len(ramsz)])
    pos_offset[1, :] = -ramsz

    states = {}
    for state, scans in (raw.get('states') or {}).items():
        try:
            states[int(state)] = (int(scans['first_scan']), int(scans['first_file']))
        except (TypeError, ValueError, KeyError):
            raise ConfigError('states: state %s needs a first_scan and a first_file, both whole numbers'
                              % state)
    if states and sorted(states) != list(range(1, max(states) + 1)):
        raise ConfigError('states: the states have to be numbered 1, 2, 3, ... with none left out '
                          '(state 0 is the first, which is not set up here), not %s' % sorted(states))

    beam_calibration_file = None
    if calibration['apply']:
        beam_calibration_file = os.path.join(output_dir, beam_calibration_file_name(analysis_stem))

    return SimpleNamespace(
        path=file_name,
        sample=raw['sample'],
        analysis_stem=analysis_stem,
        ff_dir=_path(base_dir, raw['ff_dir']),
        output_dir=output_dir,
        detector_file=_path(base_dir, raw['detector_file']),
        materials_file=_path(base_dir, materials['file']),
        material_name=materials['name'],
        ramsz=ramsz,
        num_layers=len(ramsz),
        pos_offset=pos_offset,
        beam_height=float(layers['beam_height']),
        location=seeds['location'],
        seed_margin=float(seeds['margin']),
        beam_calibration_file=beam_calibration_file,
        states=states,
        stitch=SimpleNamespace(**_section(raw, 'stitch')),
        recompile=SimpleNamespace(**_section(raw, 'recompile')),
        calibration=SimpleNamespace(**calibration),
        hexrd=SimpleNamespace(
            command=hexrd['command'],
            args=hexrd['args'],
            initial_yaml=None if hexrd['initial_yaml'] is None else _path(base_dir, hexrd['initial_yaml']),
            image_fstem=hexrd['image_fstem'],
        ),
    )


def require_for_hexrd(cfg):
    """Check that what is needed to write the hexrd configs for the later states is in the config"""
    missing = [name for name, value in (('hexrd: initial_yaml', cfg.hexrd.initial_yaml),
                                         ('hexrd: image_fstem', cfg.hexrd.image_fstem),
                                         ('states', cfg.states)) if not value]
    if missing:
        raise ConfigError('To set up the later states the config needs %s' % ', '.join(missing))
