from .config import Config


class TomographyConfig(Config):
    @property
    def img_stem(self):
        return self._cfg.get('tomography:img_stem')

    @property
    def filetype(self):
        return self._cfg.get('tomography:filetype')

    @property
    def bright(self):
        return BrightConfig(self._cfg)

    @property
    def dark(self):
        return DarkConfig(self._cfg)

    @property
    def tomo_images(self):
        return TomoImagesConfig(self._cfg)

    @property
    def num_digits(self):
        return self._cfg.get('tomography:num_digits', 6)

    @property
    def ome_range(self):
        ome_start = self._cfg.get('tomography:ome_start', 0)
        ome_end = self._cfg.get('tomography:ome_end', 359.75)
        ome_range = [(ome_start, ome_end)]
        return ome_range

    @property
    def processing(self):
        return TomoProcessingConfig(self._cfg)

    @property
    def reconstruction(self):
        return TomoReconstructionConfig(self._cfg)

    @property
    def spec(self):
        return TomoSpecConfig(self._cfg)

    @property
    def missing_slots(self):
        """Text file with the 0-based trigger slots that have no frame"""
        return self._cfg.get('tomography:missing_slots', None)

    @property
    def theta_file(self):
        """.npy file with one projection angle (deg) per frame"""
        return self._cfg.get('tomography:theta_file', None)

    @property
    def deg_per_frame(self):
        """Angular step per detector trigger (deg), overriding the spec log"""
        return self._cfg.get('tomography:deg_per_frame', None)


class TomoSpecConfig(Config):
    """The spec log and scan that describe the tomography scan"""

    @property
    def log(self):
        return self._cfg.get('tomography:spec:log', None)

    @property
    def scan(self):
        return self._cfg.get('tomography:spec:scan', None)

    @property
    def tomo_dir(self):
        """Folder with the retiga h5 files (used by the pipeline)"""
        return self._cfg.get('tomography:spec:tomo_dir', None)


class BrightConfig(Config):

    @property
    def folder(self):
        return self._cfg.get('tomography:bright:folder')

    @property
    def img_start(self):
        return self._cfg.get('tomography:bright:img_start')

    @property
    def num_imgs(self):
        return self._cfg.get('tomography:bright:num_imgs', 10)

    @property
    def filename(self):
        return self._cfg.get('tomography:bright:filename')


class DarkConfig(Config):

    @property
    def folder(self):
        return self._cfg.get('tomography:dark:folder')

    @property
    def img_start(self):
        return self._cfg.get('tomography:dark:img_start')

    @property
    def num_imgs(self):
        return self._cfg.get('tomography:dark:num_imgs', 10)

    @property
    def filename(self):
        return self._cfg.get('tomography:dark:filename')


class TomoImagesConfig(Config):

    @property
    def img_start(self):
        return self._cfg.get('tomography:tomo_images:img_start')

    @property
    def num_imgs(self):
        return self._cfg.get('tomography:tomo_images:num_images')

    @property
    def folder(self):
        return self._cfg.get('tomography:tomo_images:folder')

    @property
    def filename(self):
        return self._cfg.get('tomography:tomo_images:filename')


class TomoProcessingConfig(Config):

    @property
    def recon_thresh(self):
        return self._cfg.get('tomography:processing:recon_thresh', 0.0003)

    @property
    def noise_obj_size(self):
        return self._cfg.get('tomography:processing:noise_obj_size', 500)

    @property
    def min_hole_size(self):
        return self._cfg.get('tomography:processing:min_hole_size', 500)

    @property
    def erosion_iter(self):
        return self._cfg.get('tomography:processing:erosion_iter', 1)

    @property
    def dilation_iter(self):
        return self._cfg.get('tomography:processing:dilation_iter', 1)


class TomoReconstructionConfig(Config):

    @property
    def project_single_layer(self):
        return self._cfg.get('tomography:reconstruction:project_single_layer', False)

    @property
    def mask_filepath(self):
        return self._cfg.get('tomography:reconstruction:mask_filepath', None)

    @property
    def vertical_motor_position(self):
        return self._cfg.get(
            'tomography:reconstruction:vertical_motor_position', 0.0)

    @property
    def voxel_spacing(self):
        return self._cfg.get('tomography:reconstruction:voxel_spacing')

    @property
    def v_bnds(self):
        return self._cfg.get('tomography:reconstruction:v_bnds', [0.0, 0.0])

    @property
    def x_center(self):
        return self._cfg.get('tomography:reconstruction:x_center', 0.)

    @property
    def z_center(self):
        return self._cfg.get('tomography:reconstruction:z_center', 0.)

    @property
    def x_length(self):
        return self._cfg.get('tomography:reconstruction:x_length', 0.)

    @property
    def z_length(self):
        return self._cfg.get('tomography:reconstruction:z_length', 0.)
