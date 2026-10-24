#!/usr/bin/env python3
"""Set up the FF pipeline for a series of states

From the pipeline config (see ff_pipeline.yml) this writes, into the output directory,

    <stem>S<n>.yml        the hexrd config of each state n, with the layers' images and the seed grains
    run_state_<n>.sh      fits the layers of state n with hexrd, then recompiles them
    run_pipeline.sh       calc_lparm.py and stitch_grains.py for the first state, then run_state_<n>.sh
                          for each state in order

and does not run any of them.  It assumes that hexrd has already fit the first state (0) with a config of
your own, which is the hexrd: initial_yaml that the other configs are made from.  Each state is a first scan
and file number in the config, and its layers are the following scans and files.

To set up or redo one state, for instance after adding it to the config:

    python build_pipeline.py ff_pipeline.yml --state 3

and to do a state by hand, run its run_state_<n>.sh, or the two steps in it yourself.  The state that
follows needs the files that the recompile of this one writes, so the states have to be run in order.
"""
import argparse
import os
import shlex
import stat

from ff_config import ConfigError, load_config, require_for_hexrd
from ff_grains_io import read_detector_panels, write_yaml, yaml_file_name

HERE = os.path.dirname(os.path.abspath(__file__))


def _script(name):
    return shlex.quote(os.path.join(HERE, name))


def _write_executable(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP)


def state_script_lines(cfg, state):
    """The lines of run_state_<n>.sh"""
    hexrd = ' '.join(part for part in (cfg.hexrd.command, cfg.hexrd.args,
                                       shlex.quote(yaml_file_name(cfg.analysis_stem, state))) if part)
    return [
        '#!/bin/bash',
        '# State %d: fit the layers with hexrd, then put them back together and seed the next state' % state,
        'set -euo pipefail',
        'cd %s' % shlex.quote(cfg.output_dir),
        hexrd,
        'python %s %s --state %d' % (_script('recompile_grains.py'), shlex.quote(cfg.path), state),
    ]


def build_state(cfg, state, det_panels):
    """Write the hexrd config and the run script of one state, returns their paths"""
    first_scan, first_file = cfg.states[state]
    os.makedirs(cfg.output_dir, exist_ok=True)
    write_yaml(cfg.hexrd.initial_yaml, cfg.hexrd.image_fstem, cfg.sample, cfg.analysis_stem, state,
               cfg.num_layers, first_scan, first_file, cfg.output_dir, det_panels, cfg.location)
    script = os.path.join(cfg.output_dir, 'run_state_%d.sh' % state)
    _write_executable(script, state_script_lines(cfg, state))
    return os.path.join(cfg.output_dir, yaml_file_name(cfg.analysis_stem, state)), script


def build_pipeline(cfg, only_state=None):
    """Write the configs and scripts of every state in the config, or only_state

    Returns the paths of the files that were written.
    """
    require_for_hexrd(cfg)
    if only_state is not None and only_state not in cfg.states:
        raise ConfigError('State %d is not in the config, the states there are %s'
                          % (only_state, sorted(cfg.states)))
    det_panels = read_detector_panels(cfg.detector_file)
    states = sorted(cfg.states) if only_state is None else [only_state]

    written = []
    for state in states:
        written.extend(build_state(cfg, state, det_panels))

    if only_state is None:
        config = shlex.quote(cfg.path)
        lines = ['#!/bin/bash',
                 '# The whole pipeline, once hexrd has fit the first state (0): the calibration and the',
                 '# first master list, then each later state in order',
                 'set -euo pipefail',
                 'cd %s' % shlex.quote(cfg.output_dir),
                 'python %s %s' % (_script('calc_lparm.py'), config),
                 'python %s %s' % (_script('stitch_grains.py'), config)]
        lines += [shlex.quote(os.path.join(cfg.output_dir, 'run_state_%d.sh' % s)) for s in states]
        path = os.path.join(cfg.output_dir, 'run_pipeline.sh')
        _write_executable(path, lines)
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('config', help='the pipeline config yaml')
    parser.add_argument('--state', type=int, default=None, help='set up only this state')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for path in build_pipeline(cfg, args.state):
        print('Wrote %s' % path)


if __name__ == '__main__':
    main()
