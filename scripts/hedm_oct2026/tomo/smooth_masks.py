"""Smooth tomography masks by Gaussian-filtering their signed-distance field.

Usage: python smooth_masks.py IN.npz OUT.npz SIGMA_VOXELS [--2d]
The signed distance (voxel units, >0 inside) is filtered in 3-D (or per layer with --2d), then
thresholded at 0. OUT keeps every key of IN, replaces `mask`, and adds `mask_unsmoothed`,
`sdf_smooth` (float32, voxel units; mesh it at 0 for a smooth surface) and `smooth_sigma_voxels`.
"""
import sys
import numpy as np, scipy.ndimage as ndi

def sdf3(m):
    return ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)

def smooth(m, sigma, per_layer=False):
    if per_layer:
        s = np.stack([ndi.distance_transform_edt(x) - ndi.distance_transform_edt(~x) for x in m])
        s = np.stack([ndi.gaussian_filter(x, sigma, mode='nearest') for x in s])
    else:
        s = ndi.gaussian_filter(sdf3(m), sigma, mode='nearest')
    return s > 0, s.astype(np.float32)

if __name__ == '__main__':
    src, dst, sigma = sys.argv[1], sys.argv[2], float(sys.argv[3])
    d = dict(np.load(src)); m = d['mask']
    new, s = smooth(m, sigma, per_layer='--2d' in sys.argv)
    d.update(mask_unsmoothed=m, mask=new, sdf_smooth=s, smooth_sigma_voxels=np.array(sigma))
    np.savez_compressed(dst, **d)
    print(f'{dst}: sigma {sigma} voxels ({sigma*float(d["voxel_spacing"])*1000:.0f} um); volume {100*(new.sum()-m.sum())/m.sum():+.2f}%, voxels changed {100*(new!=m).mean():.2f}%')
