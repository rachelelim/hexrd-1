from .config import Config


class ExperimentConfig(Config):
    @property
    def beam_energy(self):
        return self._cfg.get('experiment:beam_energy')

    @property
    def beam_vertical_span(self):
        return self._cfg.get('experiment:beam_vertical_span')

    @property
    def material_name(self):
        return self._cfg.get('experiment:material_name')

    @property
    def max_tth(self):
        return self._cfg.get('experiment:max_tth', None)

    def _unit_threshold(self, name):
        temp = self._cfg.get(f'experiment:{name}', None)
        if temp is not None and not 0.0 < temp <= 1.0:
            raise RuntimeError(
                f'{name} must be None or a number between 0 and 1')
        return temp

    @property
    def comp_thresh(self):
        return self._unit_threshold('comp_thresh')

    @property
    def chi2_thresh(self):
        return self._unit_threshold('chi2_thresh')

    @property
    def omega_start(self):
        return self._cfg.get('experiment:images_and_omegas:omega_start', 0.0)

    @property
    def omega_stop(self):
        return self._cfg.get('experiment:images_and_omegas:omega_stop', 360.0)

    @property
    def omega_correction(self):
        return self._cfg.get(
            'experiment:images_and_omegas:omega_correction', 0)
