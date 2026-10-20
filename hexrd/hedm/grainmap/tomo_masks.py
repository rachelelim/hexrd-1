"""Clean up tomography masks: fill bad layers and smooth.

Both work on the signed-distance field of a mask (distance in voxels, positive inside), which
interpolates and filters much better than the binary mask. The file functions read and write the
``*_tomo_mask.npz`` files written by ``hexrd tomo reconstruct``:

* ``interpolate_mask_file`` replaces bad layers by a cubic through the signed distance of
  neighboring layers.
* ``smooth_mask_file`` Gaussian-filters the signed distance in 3-D (or per layer) and thresholds it
  at 0.
"""
import json
import logging

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


def sdf(m):
    """Signed distance of a boolean mask (voxel units, > 0 inside)."""
    return ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)


def interpolate(mask, targets, support):
    """mask (n, ny, nx) bool -> bool masks for ``targets``, from a cubic through the ``support`` layers."""
    S = np.stack([sdf(mask[i]) for i in support])
    cs = CubicSpline(support, S, axis=0)          # 4 knots = the cubic through them
    return cs(np.asarray(targets, float)) > 0


def smooth(m, sigma, per_layer=False):
    """Smooth a mask by filtering its signed distance. Returns (smoothed mask, signed distance float32)."""
    if per_layer:
        s = np.stack([sdf(x) for x in m])
        s = np.stack([ndi.gaussian_filter(x, sigma, mode='nearest') for x in s])
    else:
        s = ndi.gaussian_filter(sdf(m), sigma, mode='nearest')
    return s > 0, s.astype(np.float32)


def parse_ranges(text):
    """'15-17,28-31,63' -> [(15, 17), (28, 31), (63, 63)]"""
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out.append((int(a), int(b or a)))
    return out


def interpolate_mask_file(src, dst, ranges, support=None):
    """Fill the layer ``ranges`` [(first, last), ...] of the mask in ``src`` and write ``dst``.

    Each range is interpolated from the two layers on each side of it, unless ``support`` (four layer
    numbers) is given, which is allowed for a single range. ``dst`` keeps every key of ``src``, with
    ``mask`` replaced and ``interpolated_layers`` and ``interpolation_support`` (a JSON string, range ->
    support layers) added.
    """
    d = dict(np.load(src))
    M = d['mask']
    n = M.shape[0]
    new = M.copy()
    done, sup = [], {}
    bad = {i for a, b in ranges for i in range(a, b + 1)}
    if support is not None and len(ranges) != 1:
        raise ValueError('explicit support layers can only be given with a single range')
    for a, b in ranges:
        s = list(support) if support is not None else [a - 2, a - 1, b + 1, b + 2]
        if min(s) < 0 or max(s) >= n:
            raise ValueError(f'range {a}-{b}: needs layers {s} but the mask has 0..{n - 1}; trim this range instead')
        if set(s) & bad:
            raise ValueError(f'range {a}-{b}: support layers {s} overlap another interpolated range; merge the ranges')
        t = list(range(a, b + 1))
        new[t] = interpolate(M, t, s)
        done += t
        sup[f'{a}-{b}'] = s
        logger.info(f'interpolated layers {t} from {s}: fill {np.round(M[t].mean(axis=(1, 2)), 2).tolist()} '
                    f'-> {np.round(new[t].mean(axis=(1, 2)), 2).tolist()}')
    d['mask'] = new
    d['interpolated_layers'] = np.array(done)
    d['interpolation_support'] = np.array(json.dumps(sup))
    np.savez_compressed(dst, **d)
    logger.info(f'wrote {dst}')
    return dst


def smooth_mask_file(src, dst, sigma, per_layer=False):
    """Smooth the mask in ``src`` (sigma in voxels) and write ``dst``.

    ``dst`` keeps every key of ``src``, replaces ``mask``, and adds ``mask_unsmoothed``, ``sdf_smooth``
    (float32, voxel units; mesh it at 0 for a smooth surface) and ``smooth_sigma_voxels``.
    Returns the change in volume and the fraction of voxels changed, in percent.
    """
    d = dict(np.load(src))
    m = d['mask']
    new, s = smooth(m, sigma, per_layer=per_layer)
    d.update(mask_unsmoothed=m, mask=new, sdf_smooth=s, smooth_sigma_voxels=np.array(sigma))
    np.savez_compressed(dst, **d)
    volume = 100 * (new.sum() - m.sum()) / m.sum()
    changed = 100 * (new != m).mean()
    logger.info(f'wrote {dst}: sigma {sigma} voxels; volume {volume:+.2f}%, voxels changed {changed:.2f}%')
    return volume, changed
