import os

from .config import Config
from .multiprocessing import MultiprocessingConfig
from .NF_reconstruction import NF_ReconstructionConfig
from .NF_images import NF_ImagesConfig
from .experiment import ExperimentConfig
from .input_files import InputConfig
from .stitch import StitchConfig
from .tomography import TomographyConfig


class NFRootConfig(Config):
    @property
    def main_directory(self):
        # defaults to the working directory if unset
        return self._cfg.get('main_directory') or os.getcwd()

    @property
    def output_directory(self):
        # defaults to main_directory if unset
        return self._cfg.get('output_directory') or self.main_directory

    @property
    def analysis_name(self):
        return self._cfg.get('analysis_name', 'NF_')

    @property
    def output_plot_check(self):
        return self._cfg.get('output_plot_check', False)

    @property
    def multiprocessing(self):
        return MultiprocessingConfig(self)

    @property
    def tomography(self):
        return TomographyConfig(self)

    @property
    def stitch(self):
        return StitchConfig(self)

    @property
    def reconstruction(self):
        return NF_ReconstructionConfig(self)

    @property
    def images(self):
        return NF_ImagesConfig(self)

    @property
    def experiment(self):
        return ExperimentConfig(self)

    @property
    def input_files(self):
        return InputConfig(self)
