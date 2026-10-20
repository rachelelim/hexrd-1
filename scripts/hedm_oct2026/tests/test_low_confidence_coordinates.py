"""Tests for nfutil.generate_low_confidence_test_coordinates (no data needed)."""
import os
import sys
import unittest

import numpy as np

from hexrd.hedm.grainmap import nfutil  # noqa: E402


def reconstruction(shape, confidence=0.2):
    """A reconstruction dict with every voxel in the mask and below any threshold."""
    xs, ys, zs = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    return dict(confidence_map=np.full(shape, confidence), tomo_mask=np.ones(shape, bool),
                Xs=xs.astype(float), Ys=ys.astype(float), Zs=zs.astype(float))


class TestLowConfidenceCoordinates(unittest.TestCase):

    def count(self, shape, sparse, erode=0, threshold=0.5):
        coords, ids = nfutil.generate_low_confidence_test_coordinates(
            reconstruction(shape), confidence_threshold=threshold,
            how_sparse=sparse, errode_free_surface=erode)
        self.assertEqual(len(coords), len(ids))
        return len(ids)

    def test_not_sparse_returns_every_low_confidence_voxel(self):
        self.assertEqual(self.count((1, 8, 8), 0), 64)
        self.assertEqual(self.count((3, 8, 8), 0), 192)

    def test_single_layer_is_not_emptied_by_sparsing(self):
        # This returned no coordinates at all: the only layer has index 0, which
        # was zeroed along with every other even index
        self.assertEqual(self.count((1, 8, 8), 2), 16)
        self.assertGreater(self.count((1, 8, 8), 1), 0)

    def test_multi_layer_sparsing_is_unchanged(self):
        # every other voxel along each axis: 2 x 4 x 4 of a 4 x 8 x 8 volume
        self.assertEqual(self.count((4, 8, 8), 2), 32)
        n_some = self.count((4, 8, 8), 1)
        self.assertGreater(n_some, 32)
        self.assertLessEqual(n_some, 4*8*8)

    def test_only_low_confidence_voxels_are_returned(self):
        shape = (1, 8, 8)
        rec = reconstruction(shape)
        rec['confidence_map'][0, :4, :] = 0.9
        coords, ids = nfutil.generate_low_confidence_test_coordinates(
            rec, confidence_threshold=0.5, how_sparse=0, errode_free_surface=0)
        self.assertEqual(len(ids), 32)

    def test_unique_ids(self):
        _, ids = nfutil.generate_low_confidence_test_coordinates(
            reconstruction((2, 6, 6)), 0.5, 0, 0)
        self.assertEqual(len(np.unique(ids)), len(ids))


if __name__ == '__main__':
    unittest.main()
