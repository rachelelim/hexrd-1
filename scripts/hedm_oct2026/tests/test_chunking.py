"""Tests for nfutil.processor_functions.make_chunks."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfutil.processor_functions import make_chunks  # noqa: E402


def legacy_chunks(n, ncpus, chunk_size):
    """The loop that was previously copied into each handler."""
    if chunk_size == -1:
        chunk_size = int(np.ceil(n/ncpus))
    num_chunks = int(np.ceil(n/chunk_size))
    starts = np.zeros(num_chunks, dtype=int)
    stops = np.zeros(num_chunks, dtype=int)
    for i in np.arange(num_chunks):
        starts[i] = i*chunk_size
        stops[i] = i*chunk_size + chunk_size
        if stops[i] >= n:
            stops[i] = n
    return np.arange(num_chunks), starts, stops, chunk_size


class TestMakeChunks(unittest.TestCase):

    def test_matches_legacy_loop(self):
        for n in (1, 2, 7, 184, 10202, 1444):
            for ncpus in (1, 2, 9, 16):
                for chunk_size in (-1, 1, 5, 1000):
                    new = make_chunks(n, ncpus, chunk_size)
                    old = legacy_chunks(n, ncpus, chunk_size)
                    for a, b in zip(new, old):
                        np.testing.assert_array_equal(a, b)

    def test_covers_every_item_once(self):
        _, starts, stops, _ = make_chunks(10202, 9)
        covered = np.concatenate(
            [np.arange(a, b) for a, b in zip(starts, stops)])
        np.testing.assert_array_equal(covered, np.arange(10202))


if __name__ == '__main__':
    unittest.main()
