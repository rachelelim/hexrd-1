"""Fill bad layers of a tomo mask by cubic interpolation of signed-distance maps.

Usage: python interp_layers.py MASK.npz OUT.npz  bad_first bad_last  [support layers ...]
e.g.   python interp_layers.py diamond-1a_S0_L1b_tomo_mask.npz fixed.npz 31 34 29 30 35 36
"""
import sys
import numpy as np, scipy.ndimage as ndi
from scipy.interpolate import CubicSpline

def sdf(m):
    return ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)   # >0 inside, voxel units

def interpolate(mask, targets, support):
    """mask (n,ny,nx) bool -> bool masks for `targets`, from cubic through `support` layers."""
    S = np.stack([sdf(mask[i]) for i in support])
    cs = CubicSpline(support, S, axis=0)          # 4 knots = the cubic through them
    return cs(np.asarray(targets, float)) > 0

if __name__ == '__main__':
    src, dst, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    support = [int(x) for x in sys.argv[5:]]
    d = dict(np.load(src)); m = d['mask']
    targets = list(range(a, b + 1))
    new = interpolate(m, targets, support)
    print('interpolated layers', targets, 'from', support, '| fill before', np.round(m[targets].mean(axis=(1, 2)), 2).tolist(),
          'after', np.round(new.mean(axis=(1, 2)), 2).tolist())
    m2 = m.copy(); m2[targets] = new
    d['mask'] = m2; d['interpolated_layers'] = np.array(targets); d['interpolation_support'] = np.array(support)
    np.savez_compressed(dst, **d); print('wrote', dst)
