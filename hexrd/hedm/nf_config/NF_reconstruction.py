from .config import Config


class NF_ReconstructionConfig(Config):
    @property
    def tomography(self):
        prefix = 'NF_reconstruction:tomography:'
        if self._cfg.get(prefix + 'mask_filepath', None) is None:
            return None
        # vertical_motor_position, or spec_log with nf_scan, give the vertical offset of the mask;
        # see nfutil.resolve_vertical_motor_position
        return {name: self._cfg.get(prefix + name, None)
                for name in ('mask_filepath',
                             'vertical_motor_position',
                             'use_single_layer',
                             'spec_log',
                             'nf_scan')}

    @property
    def cross_sectional_dimensions(self):
        return self._cfg.get('NF_reconstruction:cross_sectional_dimensions', 1)

    @property
    def voxel_spacing(self):
        return self._cfg.get('NF_reconstruction:voxel_spacing')

    @property
    def x_center(self):
        return self._cfg.get('NF_reconstruction:x_center', 0.)

    @property
    def z_center(self):
        return self._cfg.get('NF_reconstruction:z_center', 0.)

    @property
    def x_length(self):
        return self._cfg.get('NF_reconstruction:x_length', 1.)

    @property
    def z_length(self):
        return self._cfg.get('NF_reconstruction:z_length', 1.)

    @property
    def desired_vertical_span(self):
        # defaults to the beam vertical span, which is only required if this is not set
        span = self._cfg.get('NF_reconstruction:desired_vertical_span', None)
        if span is None:
            span = self._cfg.get('experiment:beam_vertical_span')
        return span

    @property
    def beam_stop(self):
        key = self._cfg.get(
            'NF_reconstruction:beam_stop:beam_stop_filepath', None)
        if key is not None:
            return key
        prefix = 'NF_reconstruction:beam_stop:'
        return [self._cfg.get(prefix + 'beam_stop_vertical_center', 0.0),
                self._cfg.get(prefix + 'beam_stop_vertical_span', 0.3)]

    @property
    def misorientation(self):
        if self._cfg.get(
                'experiment:misorientation:use_misorientation', False) is True:
            return dict(
                misorientation_bnd=self._cfg.get(
                    'experiment:misorientation:bound', 0.0),
                misorientation_spacing=self._cfg.get(
                    'experiment:misorientation:spacing', 0.25))
        return None

    @property
    def missing_grains(self):
        prefix = 'NF_reconstruction:missing_grains'
        if self._cfg.get(prefix, None) is None:
            return None
        return {name: self._cfg.get(f'{prefix}:{name}')
                for name in ('reconstructed_data_path',
                             'ori_grid_spacing',
                             'confidence_threshold',
                             'low_confidence_sparsing',
                             'errode_free_surface',
                             'coord_cutoff_scale',
                             'iter_cutoff',
                             're_run_and_save')}
