import os

from .config import Config


class StitchConfig(Config):
    @property
    def layer_directory(self):
        # defaults to output_directory if unset
        return (self._cfg.get('stitch:layer_directory', None)
                or self._cfg.output_directory)

    @property
    def layer_stem(self):
        stem = self._cfg.get('stitch:layer_stem')
        if '%d' not in stem:
            raise RuntimeError(
                'stitch:layer_stem must contain %d for the layer number')
        return stem

    @property
    def layer_motor_positions(self):
        # One per layer, in stacking order. Each is that layer's own vertical sample stage
        # position (ramsz, mm), which the stitching subtracts from the layer's heights. It is
        # not the vertical offset of the layer from the spec log (ramsz_nf - ramsz_tomography)
        return list(self._cfg.get('stitch:layer_motor_positions'))

    @property
    def nlayers(self):
        return len(self.layer_motor_positions)

    @property
    def layer_files(self):
        return [
            os.path.join(self.layer_directory,
                         self.layer_stem % i + '_grain_map_data.npz')
            for i in range(self.nlayers)
        ]

    @property
    def tomography_mask(self):
        return self._cfg.get('stitch:tomography_mask_filepath', None)

    @property
    def voxel_spacing(self):
        return self._cfg.get('stitch:voxel_spacing')

    @property
    def overlap(self):
        return self._cfg.get('stitch:overlap', 0.0)

    @property
    def overlap_voxels(self):
        return int(round(self.overlap / self.voxel_spacing))

    @property
    def confidence_threshold(self):
        return self._cfg.get('stitch:confidence_threshold', 0.22)

    @property
    def output_stem(self):
        # defaults to <analysis_name>_stitched if unset
        return (self._cfg.get('stitch:output_stem', None)
                or self._cfg.analysis_name + '_stitched')
