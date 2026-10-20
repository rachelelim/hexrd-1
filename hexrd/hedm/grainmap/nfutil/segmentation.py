"""Grains from a refined orientation field, and the misorientation of each voxel from its grain's average.

After refinement every voxel has its own orientation, so there are no grains until voxels are
grouped. ``segment_orientation_field`` joins neighboring voxels whose orientations differ by less
than a threshold (connected regions), ``grain_average_orientations`` averages the orientations of
each region, and ``misorientation_from_average`` gives each voxel's distance from its region's
average, the grain reference orientation deviation.

All misorientations count crystal symmetry: orientations that differ by a symmetry operation of the
crystal are the same orientation. The operators act on the crystal side of the orientation, as they
do for the orientations in this package (``q_sym`` is ``plane_data.q_sym``).

contributing authors: lim37
"""

import numpy as np

from hexrd import rotations
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .spatial_adaptive import neighbor_table

_CONJUGATE = np.array([1., -1., -1., -1.])


def quaternions_of_exp_maps(exp_maps):
    """(n, 3) exponential maps to (n, 4) unit quaternions (w, x, y, z)."""
    return np.atleast_2d(rotations.quatOfExpMap(np.asarray(exp_maps, float).reshape(-1, 3).T).T)


def exp_maps_of_quaternions(quats):
    """(n, 4) unit quaternions to (n, 3) exponential maps."""
    return np.atleast_2d(rotations.expMapOfQuat(np.asarray(quats, float).reshape(-1, 4).T).T)


def _multiply(a, b):
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw], axis=1)


def misorientation_pairs(quats_a, quats_b, q_sym):
    """Misorientation in degrees between quats_a[i] and quats_b[i], with crystal symmetry.

    quats_* are (n, 4); q_sym is (4, n_sym), the crystal symmetry operators.
    """
    relative = _multiply(quats_a*_CONJUGATE, quats_b)
    # real part of relative * symmetry operator, for every operator
    w = np.abs(relative[:, [0]]*q_sym[0][None, :] - relative[:, 1:] @ q_sym[1:])
    return np.degrees(2.*np.arccos(np.minimum(1., w.max(axis=1))))


def segment_orientation_field(exp_maps, mask, q_sym, threshold_deg=2.0, connectivity=6):
    """Group voxels into regions of nearly the same orientation.

    Neighboring voxels (``connectivity`` as in ``neighbor_table``) are joined if their orientations
    differ by less than ``threshold_deg``, and a region is a connected set of joined voxels.
    ``exp_maps`` are (n, 3), over the masked voxels in the order of ``mask[mask]``.

    Returns ``labels`` (n,), numbered from 0 by decreasing size.
    """
    exp_maps = np.asarray(exp_maps, float)
    n = len(exp_maps)
    neighbors = neighbor_table(mask, connectivity)
    i, k = np.where(neighbors >= 0)
    j = neighbors[i, k]
    keep = i < j                     # each pair once
    i, j = i[keep], j[keep]
    quats = quaternions_of_exp_maps(exp_maps)
    joined = misorientation_pairs(quats[i], quats[j], q_sym) < threshold_deg
    graph = coo_matrix((np.ones(joined.sum()), (i[joined], j[joined])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    # renumber by decreasing size
    order = np.argsort(-np.bincount(labels), kind='stable')
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    return renumber[labels]


def neighbor_misorientations(exp_maps, mask, q_sym, connectivity=6):
    """Misorientation (degrees) of every pair of neighboring voxels, each pair once."""
    neighbors = neighbor_table(mask, connectivity)
    i, k = np.where(neighbors >= 0)
    j = neighbors[i, k]
    keep = i < j
    quats = quaternions_of_exp_maps(exp_maps)
    return misorientation_pairs(quats[i[keep]], quats[j[keep]], q_sym)


def _average_quaternion(quats, weights, q_sym):
    """Weighted average of orientations that may be symmetry-equivalent copies of each other."""
    reference = quats[np.argmax(weights)]
    for _ in range(2):
        # the symmetry-equivalent of each orientation that is closest to the reference
        best = None
        for s in q_sym.T:
            candidate = _multiply(quats, np.broadcast_to(s, quats.shape))
            dot = candidate @ reference
            if best is None:
                best, best_dot = candidate.copy(), dot.copy()
            else:
                better = np.abs(dot) > np.abs(best_dot)
                best[better] = candidate[better]
                best_dot[better] = dot[better]
        best *= np.where(best_dot < 0, -1., 1.)[:, None]
        reference = (weights[:, None]*best).sum(axis=0)
        reference /= np.linalg.norm(reference)
    return reference


def grain_average_orientations(exp_maps, labels, q_sym, weights=None):
    """Average orientation of every region, as (n_labels, 3) exponential maps.

    The orientations of a region are first turned into the symmetry-equivalents closest to the
    region's best-weighted voxel, so a region that mixes equivalent orientations is averaged
    correctly. ``weights`` (n,) default to 1, and could be the confidence of each voxel.
    """
    exp_maps = np.asarray(exp_maps, float)
    labels = np.asarray(labels)
    weights = np.ones(len(exp_maps)) if weights is None else np.asarray(weights, float)
    quats = quaternions_of_exp_maps(exp_maps)
    averages = np.zeros((labels.max() + 1, 4))
    for label in range(labels.max() + 1):
        members = np.where(labels == label)[0]
        averages[label] = _average_quaternion(quats[members], weights[members], q_sym)
    return exp_maps_of_quaternions(averages)


def misorientation_from_average(exp_maps, labels, averages, q_sym):
    """Misorientation (degrees) of every voxel from the average orientation of its region."""
    quats = quaternions_of_exp_maps(exp_maps)
    average_quats = quaternions_of_exp_maps(averages)
    return misorientation_pairs(average_quats[np.asarray(labels)], quats, q_sym)
