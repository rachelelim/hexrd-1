from .config import Config


class NF_ImagesConfig(Config):
    @property
    def loading(self):
        return LoadingConfig(self._cfg)

    @property
    def processing(self):
        return ProcessingConfig(self._cfg)


class LoadingConfig(Config):
    @property
    def load_style(self):
        return self._cfg.get('NF_images:image_loading:style')

    @property
    def sample_raw_data_folder(self):
        return self._cfg.get('NF_images:sample_raw_data_folder')

    @property
    def json_and_par_starter(self):
        return self._cfg.get('NF_images:image_loading:json_and_par_starter')

    @property
    def vertical_motor_name(self):
        return self._cfg.get('NF_images:image_loading:vertical_motor_name')

    @property
    def target_vertical_position(self):
        return self._cfg.get(
            'NF_images:image_loading:target_vertical_position')

    @property
    def omega_starts(self):
        return self._cfg.get('NF_images:image_loading:omega_starts')

    @property
    def omega_stops(self):
        return self._cfg.get('NF_images:image_loading:omega_stops')

    @property
    def nframes(self):
        return self._cfg.get(
            'NF_images:image_loading:nframes_per_folder', 1440)

    @property
    def data_folders(self):
        return self._cfg.get('NF_images:image_loading:data_folders')

    # Used by the 'direct' load style only
    @property
    def stem(self):
        return self._cfg.get('NF_images:image_loading:stem')

    @property
    def num_digits(self):
        return self._cfg.get('NF_images:image_loading:stem_digits')

    @property
    def img_start(self):
        return self._cfg.get('NF_images:image_loading:img_start')

    @property
    def goodstarts(self):
        return self._cfg.get('NF_images:image_loading:goodstarts')


class ProcessingConfig(Config):

    @property
    def omega_kernel_size(self):
        return self._cfg.get('NF_images:processing:omega_kernel_size', 50)

    @property
    def threshold(self):
        return self._cfg.get('NF_images:processing:threshold', 0)

    @property
    def dilate_omega(self):
        return self._cfg.get('NF_images:processing:dilate_omega', 0)

    @property
    def method(self):
        """Filter parameters list consumed by nfutil.filter_and_binarize_images

        [remove_small_objects, small_object_size, routine_choice,
         *routine-specific parameters]
        """
        def get(name, default):
            return self._cfg.get(f'NF_images:processing:{name}', default)

        key = get('routine_choice', 0)
        small_object_size = get('small_object_size', 0) or 0
        remove_small_objects = int(small_object_size > 0)

        if key == 0:
            # Gaussian blur
            parms = [get('sigma', 1.0),
                     get('gaussian_binarization_threshold', 3)]
        elif key == 1:
            # Erosion/Dilation
            parms = [get('num_errosions', 3),
                     get('num_dilations', 2),
                     get('erosion_dilation_binarization_threshold', 5)]
        elif key == 2:
            # Non-Local Means
            parms = [get('patch_size', 3),
                     get('patch_distance', 5),
                     get('NLM_binarization_threshold', 10)]
        else:
            raise RuntimeError(
                f'routine_choice must be 0, 1 or 2, got {key}')
        return [remove_small_objects, small_object_size, key, *parms]
