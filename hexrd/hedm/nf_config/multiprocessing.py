import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')


class MultiprocessingConfig(Config):
    @property
    def num_cpus(self):
        # determine number of processes to run in parallel
        multiproc = self.get('multiprocessing:num_cpus', default='half')
        ncpus = os.cpu_count()
        if multiproc == 'all':
            res = ncpus
        elif multiproc == 'half':
            temp = int(ncpus // 2)
            res = temp if temp else 1
        elif isinstance(multiproc, int):
            if multiproc >= 0:
                if multiproc > ncpus:
                    logger.warning(
                        'Requested %s processes, %d available',
                        multiproc, ncpus
                    )
                    res = ncpus
                else:
                    res = multiproc if multiproc else 1
            else:
                temp = ncpus + multiproc
                if temp < 1:
                    logger.warning(
                        'Cannot use less than 1 process, requested %d of %d',
                        temp, ncpus
                    )
                    res = 1
                else:
                    res = temp
        else:
            temp = ncpus - 1
            logger.warning(
                "Invalid value %s for multiprocessing",
                multiproc
            )
            res = temp
        return res

    @num_cpus.setter
    def num_cpus(self, val):
        if val in ('half', 'all', -1):
            self.set('multiprocessing:num_cpus', val)
        elif isinstance(val, int) and 0 <= val <= os.cpu_count():
            self.set('multiprocessing:num_cpus', val)
        else:
            raise RuntimeError(
                '"num_cpus": must be 1:%d, got %s'
                % (os.cpu_count(), val)
            )

    @property
    def check(self):
        return self._cfg.get('multiprocessing:check', None)

    @property
    def limit(self):
        return self._cfg.get('multiprocessing:limit', None)

    @property
    def generate(self):
        return self._cfg.get('multiprocessing:generate', None)

    @property
    def chunk_size(self):
        return self._cfg.get('multiprocessing:chunk_size', -1)
