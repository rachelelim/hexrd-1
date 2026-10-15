from .config import Config


class InputConfig(Config):
    @property
    def detector_file(self):
        return self._cfg.get('input_files:detector_file')

    @property
    def materials_file(self):
        return self._cfg.get('input_files:materials_file', None)

    @property
    def grains_out_file(self):
        return self._cfg.get('input_files:grains_out_file', None)
