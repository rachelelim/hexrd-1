"""Merge the tomography masks of one sample, scanned at several heights, into one tall mask.

A sample taller than the field of view of the detector is scanned at several heights, one
``hexrd tomo reconstruct`` mask for each, which cover overlapping ranges.  ``merge_mask_files`` puts them
together:

* the vertical position of each mask, relative to the reference mask, gives how far its heights are shifted
  (see ``resolve_offsets``): ``ramsz of the reference scan - ramsz of the scan``, the same way the NF
  reconstruction ties a mask to an NF layer, see ``resolve_vertical_motor_position``;
* each mask is registered to its neighbor, working outwards from the reference, because the sample and
  the stages do not return to exactly the same place: the shift in the plane of the layers is found from
  the slices that the two masks have in common (``register_masks``);
* the masks are resampled onto one regular grid of heights, spaced like the voxels, that starts on the grid
  of the reference mask, and where masks overlap their signed-distance fields are blended, weighted by how
  far the height is from the end of each mask, so there is no step at the joins (``merge_masks``).

The merged file has the keys of a ``*_tomo_mask.npz`` (``mask``, ``Xs``, ``Ys``, ``Zs``, ``voxel_spacing``),
heights in the frame of the reference scan, and ``ramsz`` of the reference scan when it is known, so that
the NF reconstruction can place it with the spec log.  It also has ``merge_offsets``, ``merge_shifts_voxels``,
``merge_spans_mm`` and ``merged_from``.

``plot_merge`` draws vertical cross sections of the merged mask before and after the registration, to check it.
"""
import json
import logging
import os

import numpy as np
import scipy.ndimage as ndi
from skimage.registration import phase_cross_correlation

from hexrd.hedm.grainmap.tomo_masks import sdf
from hexrd.hedm.grainmap.tomo_spec import ramsz_of_scan

logger = logging.getLogger(__name__)

_GRID_TOL = 1e-3  # a height this close to a slice, in slices, is on it
_RAMSZ_TOL = 1e-3  # mm, for the offsets and the ramsz of a mask to agree


def mask_heights(mask, offset=0.0):
    """Heights (mm) of the layers of a mask file's contents, shifted by ``offset``"""
    return np.asarray(mask['Ys'][:, 0, 0], float) + offset


def check_masks(masks):
    """Check that masks can be merged: the same grid in the plane of the layers and in height"""
    if len(masks) < 2:
        raise ValueError('merging needs at least two masks')
    first = masks[0]
    vs = float(first['voxel_spacing'])
    for i, m in enumerate(masks):
        if m['mask'].ndim != 3:
            raise ValueError(f'mask {i} is not 3-D')
        if float(m['voxel_spacing']) != vs:
            raise ValueError(f'mask {i} has voxel spacing {float(m["voxel_spacing"])}, mask 0 has {vs}')
        if m['mask'].shape[1:] != first['mask'].shape[1:]:
            raise ValueError(f'mask {i} is {m["mask"].shape[1:]} in the plane of the layers, '
                             f'mask 0 is {first["mask"].shape[1:]}')
        for axis in ('Xs', 'Zs'):
            if not np.allclose(m[axis][0], first[axis][0], atol=1e-6, rtol=0):
                raise ValueError(f'mask {i} is on a different grid in {axis[0]} than mask 0')
        step = np.diff(mask_heights(m))
        if len(step) and not np.allclose(step, vs, rtol=0, atol=0.01 * vs):
            raise ValueError(f'the layers of mask {i} are not {vs} mm apart')


def resolve_offsets(masks, offsets=None, ramsz=None, spec_log=None, tomo_scans=None, reference=None,
                    names=None):
    """How far to shift the heights of each mask, which mask is the reference, and the ramsz of the reference

    The vertical position of a mask relative to the reference is found, in order of preference, from

    1. ``offsets``, given directly (mm, added to the heights of each mask);
    2. ``ramsz``, the vertical stage position of the tomography scan of each mask;
    3. the ``ramsz`` that each mask records, when all of them do;
    4. the ``spec_log`` and the spec scan number of each mask, ``tomo_scans``.

    The shift of a mask is ``ramsz of the reference - ramsz of the mask``.  The metadata of the masks is not
    always right, so what is given directly is used as it is.  The reference is ``reference`` (an index) or,
    by default, the mask closest to the middle: the one with an offset of 0 when the offsets are given, or
    the median ramsz.

    Returns ``(offsets, reference, reference_ramsz)``, with ``reference_ramsz`` None when it is not known.
    """
    n = len(masks)
    names = names or [f'mask {i}' for i in range(n)]
    recorded = [float(m['ramsz']) if 'ramsz' in m and np.isfinite(float(m['ramsz'])) else None for m in masks]

    if offsets is not None:
        offsets = np.asarray(offsets, float)
        if offsets.shape != (n,):
            raise ValueError(f'{len(offsets)} offsets given for {n} masks')
        ref = int(np.argmin(np.abs(offsets))) if reference is None else reference
        # the ramsz of the reference is only passed on if it agrees with the offsets, since it is not
        # always right
        known = None
        if ramsz is not None:
            known = float(ramsz[ref])
        elif all(r is not None for r in recorded):
            implied = recorded[ref] - np.array(recorded)
            if np.allclose(implied, offsets, atol=_RAMSZ_TOL):
                known = recorded[ref]
            else:
                logger.warning(f'the ramsz that the masks record ({recorded}) do not agree with the offsets '
                               f'{offsets.tolist()}; the merged mask will not record a ramsz')
        return offsets, ref, known

    if ramsz is not None:
        source = 'the ramsz given'
        ramsz = np.asarray(ramsz, float)
    elif all(r is not None for r in recorded):
        source = 'the ramsz recorded in the masks'
        ramsz = np.array(recorded)
    elif spec_log is not None and tomo_scans is not None:
        source = f'the spec log {spec_log}'
        found = [ramsz_of_scan(spec_log, int(s)) for s in tomo_scans]
        if any(f is None for f in found):
            missing = [int(s) for s, f in zip(tomo_scans, found) if f is None]
            raise ValueError(f'{spec_log} has no vertical position (ramsz) for scan(s) {missing}')
        ramsz = np.array(found, float)
    else:
        raise ValueError('the vertical position of the masks is not known: give offsets, ramsz, a spec log with '
                         'the tomography scan of each mask, or use masks that record their ramsz '
                         f'({", ".join(n for n, r in zip(names, recorded) if r is None)} do not)')
    if ramsz.shape != (n,):
        raise ValueError(f'{len(ramsz)} ramsz values for {n} masks')
    ref = int(np.argsort(ramsz)[(n - 1) // 2]) if reference is None else reference
    offsets = ramsz[ref] - ramsz
    logger.info(f'Offsets {np.round(offsets, 4).tolist()} mm from {source}')
    return offsets, ref, float(ramsz[ref])


def _overlap_pairs(ref_heights, mov_heights, vs, count):
    """Index pairs (in the reference, in the moving mask) of up to ``count`` slices that are at the same height"""
    lo = max(ref_heights[0], mov_heights[0])
    hi = min(ref_heights[-1], mov_heights[-1])
    if hi < lo - 0.5 * vs:
        return []
    ys = mov_heights[(mov_heights >= lo - 0.5 * vs) & (mov_heights <= hi + 0.5 * vs)]
    if len(ys) > count:
        ys = ys[np.linspace(0, len(ys) - 1, count).round().astype(int)]
    pairs = []
    for y in ys:
        j = int(np.argmin(np.abs(mov_heights - y)))
        i = int(np.argmin(np.abs(ref_heights - y)))
        if abs(ref_heights[i] - y) <= 0.5 * vs:
            pairs.append((i, j))
    return pairs


def register_masks(ref_mask, ref_heights, mov_mask, mov_heights, vs, count=5, max_shift=10.0, name='mask'):
    """Shift, in voxels in the plane of the layers, that registers ``mov_mask`` with ``ref_mask``

    Found from up to ``count`` slices that the masks have at the same height, as the median of the
    shifts of each, by phase correlation to a tenth of a voxel.  Returns ``(shift, spread, used)``,
    the spread being the largest difference of one slice from the median.
    """
    shifts = []
    for i, j in _overlap_pairs(ref_heights, mov_heights, vs, count):
        a, b = ref_mask[i].astype(float), mov_mask[j].astype(float)
        if a.any() and b.any():
            shifts.append(phase_cross_correlation(a, b, upsample_factor=10, normalization=None)[0])
    if not shifts:
        overlap = min(ref_heights[-1], mov_heights[-1]) - max(ref_heights[0], mov_heights[0])
        raise ValueError(f'{name} has no usable slices in common with the mask it is registered to (they overlap by '
                         f'{max(0.0, overlap):.3g} mm, and the slices there have to be filled). '
                         'Check the offsets, or turn the registration off')
    shifts = np.array(shifts)
    shift = np.median(shifts, axis=0)
    spread = float(np.abs(shifts - shift).max())
    if np.abs(shift).max() > max_shift:
        raise ValueError(f'{name}: the shift that registers it is {shift.round(2).tolist()} voxels, more than '
                         f'{max_shift}. Check the offsets, which decide which slices are compared')
    if spread > 1.0:
        logger.warning(f'{name}: the shifts from the {len(shifts)} overlapping slices differ by up to '
                       f'{spread:.2f} voxels, so the registration is uncertain')
    return shift, spread, len(shifts)


def shift_mask(mask, shift):
    """A boolean mask moved by ``shift`` voxels in the plane of its layers, without growing it"""
    if not np.any(np.abs(shift) > 1e-6):
        return mask.copy()
    out = np.empty_like(mask)
    for k in range(mask.shape[0]):
        # linear interpolation, then back to a mask at half, so that the edges do not creep outwards
        out[k] = ndi.shift(mask[k].astype(np.float32), shift, order=1, mode='nearest') > 0.5
    return out


def _slice_sdf(mask, heights, y, vs):
    """Signed distance of the slice of a mask at height y, between its layers if it is not on one"""
    p = (y - heights[0]) / vs
    k = min(max(int(np.floor(p)), 0), len(heights) - 2)
    frac = p - k
    if abs(frac) < _GRID_TOL:
        return sdf(mask[k])
    if abs(frac - 1) < _GRID_TOL:
        return sdf(mask[k + 1])
    return (1 - frac) * sdf(mask[k]) + frac * sdf(mask[k + 1])


def merge_masks(masks, offsets, reference=None, register=True, register_slices=5, max_shift=10.0,
                names=None):
    """Merge masks (dicts with mask, Xs, Ys, Zs and voxel_spacing) shifted in height by ``offsets``

    Returns a dict with the merged ``mask``, ``Xs``, ``Ys``, ``Zs``, ``voxel_spacing`` and, for the
    record, ``merge_offsets`` and ``merge_shifts_voxels`` (the shift that registered each mask, in
    voxels in the plane of the layers, 0 for the reference).
    """
    check_masks(masks)
    n = len(masks)
    names = names or [f'mask {i}' for i in range(n)]
    offsets = np.asarray(offsets, float)
    ref = int(np.argmin(np.abs(offsets))) if reference is None else reference
    vs = float(masks[0]['voxel_spacing'])
    heights = [mask_heights(m, o) for m, o in zip(masks, offsets)]
    stacks = [np.asarray(m['mask'], bool) for m in masks]
    shifts = np.zeros([n, 2])

    if register:
        # outwards from the reference, each mask to its neighbor on the side of the reference, which is
        # already registered
        order = np.argsort([h.mean() for h in heights])
        pos = int(np.where(order == ref)[0][0])
        for step, rng in ((-1, range(pos - 1, -1, -1)), (1, range(pos + 1, n))):
            for p in rng:
                i, nb = order[p], order[p - step]
                shifts[i], spread, used = register_masks(
                    stacks[nb], heights[nb], stacks[i], heights[i], vs, register_slices, max_shift, names[i])
                logger.info(f'{names[i]} registered to {names[nb]}: shift {shifts[i].round(2).tolist()} voxels '
                            f'(from {used} slices, spread {spread:.2f})')
                stacks[i] = shift_mask(stacks[i], shifts[i])

    # the masks have to reach each other, or there are heights that nothing says anything about
    ordered = sorted(range(n), key=lambda i: heights[i][0])
    for a, b in zip(ordered[:-1], ordered[1:]):
        gap = heights[b][0] - heights[a][-1]
        if gap > vs * (1 + _GRID_TOL):
            raise ValueError(f'there is a gap of {gap:.4g} mm between {names[a]} (up to {heights[a][-1]:.4g} mm) and '
                             f'{names[b]} (from {heights[b][0]:.4g} mm): the masks do not reach each other. '
                             'Check the offsets')

    # the grid of heights, on that of the reference
    lo = min(h[0] for h in heights)
    hi = max(h[-1] for h in heights)
    k_lo = int(np.ceil((lo - heights[ref][0]) / vs - _GRID_TOL))
    k_hi = int(np.floor((hi - heights[ref][0]) / vs + _GRID_TOL))
    grid = heights[ref][0] + vs * np.arange(k_lo, k_hi + 1)

    merged = np.zeros([len(grid)] + list(stacks[0].shape[1:]), bool)
    for g, y in enumerate(grid):
        covering = [i for i in range(n) if heights[i][0] - 1e-6 <= y <= heights[i][-1] + 1e-6]
        if len(covering) == 1:
            i = covering[0]
            p = (y - heights[i][0]) / vs
            if abs(p - round(p)) < _GRID_TOL:
                merged[g] = stacks[i][int(round(p))]  # on a layer of the mask: as it is
            else:
                merged[g] = _slice_sdf(stacks[i], heights[i], y, vs) > 0
        else:
            # blend the signed distances, weighted by how far y is from the end of each mask
            total = 0.0
            weights = 0.0
            for i in covering:
                w = min(y - heights[i][0], heights[i][-1] - y) + vs
                total = total + w * _slice_sdf(stacks[i], heights[i], y, vs)
                weights += w
            merged[g] = total / weights > 0

    shape = merged.shape
    Xs = np.broadcast_to(masks[ref]['Xs'][0], shape).copy()
    Zs = np.broadcast_to(masks[ref]['Zs'][0], shape).copy()
    Ys = np.broadcast_to(grid[:, None, None], shape).copy()
    logger.info(f'merged {n} masks into {shape}: heights {grid[0]:.4f} to {grid[-1]:.4f} mm, '
                f'{len(grid)} layers of {vs} mm')
    return {'mask': merged, 'Xs': Xs, 'Ys': Ys, 'Zs': Zs, 'voxel_spacing': np.array(vs),
            'merge_offsets': offsets, 'merge_shifts_voxels': shifts,
            'merge_spans_mm': np.array([[h[0], h[-1]] for h in heights])}


def _lateral(coords_x, coords_z, index, axis):
    """The section at ``index`` of ``axis`` (1 or 2) of the layers: the coordinates and name of the direction
    it runs along, and the name and value of the one it is at"""
    sel = (0, index, slice(None)) if axis == 1 else (0, slice(None), index)
    options = {'x': coords_x[sel], 'z': coords_z[sel]}
    along = max(options, key=lambda k: np.ptp(options[k]))
    at = 'x' if along == 'z' else 'z'
    return options[along], along, at, float(options[at][0])


def plot_merge(before, after, names, path, section=None):
    """Draw vertical cross sections of the merged mask before and after the registration

    ``before`` and ``after`` are the results of ``merge_masks`` without and with registration.  There is one row
    for each of two sections through the sample (the middle of the sample along each of the two directions in
    the plane of the layers, or the voxels in ``section``, (index along axis 1, index along axis 2)); in each,
    the mask before, the mask after, and where they differ, with the heights where the masks overlap shaded.
    The last panel is the fraction of each layer that is sample, which has a step at a join if the masks do not
    line up there.  Written to ``path``.
    """
    from matplotlib.figure import Figure

    m_before, m_after = before['mask'], after['mask']
    y = after['Ys'][:, 0, 0]
    vs = float(after['voxel_spacing'])
    spans = after['merge_spans_mm']
    if section is None:
        occupied = np.argwhere(m_after)
        section = tuple(int(round(v)) for v in occupied[:, 1:].mean(axis=0))
    order = np.argsort(spans[:, 0])
    overlaps = [(max(spans[a, 0], spans[b, 0]), min(spans[a, 1], spans[b, 1])) for a, b in zip(order[:-1], order[1:])]
    colors = [f'C{i}' for i in range(len(spans))]

    fig = Figure(figsize=(15, 11))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.25, hspace=0.2)
    heights_extent = (y[0] - vs / 2, y[-1] + vs / 2)
    axes = []
    for row, axis in enumerate((1, 2)):
        index = section[axis - 1]
        h, h_name, at_name, at_value = _lateral(after['Xs'], after['Zs'], index, axis)
        cut_before = np.take(m_before, index, axis=axis)
        cut_after = np.take(m_after, index, axis=axis)
        # layers (height) up the page, along the section across; for imshow the first axis is up the page
        extent = (h.min() - vs / 2, h.max() + vs / 2, heights_extent[0], heights_extent[1])
        both = np.where(cut_before & cut_after, 0.75, 1.0)
        rgb = np.stack([both, both, both], axis=-1)
        rgb[cut_before & ~cut_after] = (0.85, 0.1, 0.1)  # only before the registration
        rgb[~cut_before & cut_after] = (0.1, 0.3, 0.9)  # only after it
        panels = [(cut_before, 'before the registration', 'gray_r'),
                  (cut_after, 'after the registration', 'gray_r'),
                  (rgb, 'where they differ\n(red: only before, blue: only after, grey: both)', None)]
        for col, (img, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(img, origin='lower', extent=extent, aspect='equal', cmap=cmap, interpolation='nearest')
            for lo, hi in overlaps:
                ax.axhspan(lo, hi, color='tab:orange', alpha=0.12, lw=0)
            ax.set_title(title if row == 0 else '', fontsize=9)
            ax.set_xlabel(f'{h_name} (mm)')
            if col == 0:
                ax.set_ylabel(f'height (mm)\nsection at {at_name} = {at_value:.3f} mm')
            else:
                ax.set_yticklabels([])
            axes.append(ax)

    ax = fig.add_subplot(grid[:, 3])
    for i, ((lo, hi), name) in enumerate(zip(spans, names)):
        ax.axhspan(lo, hi, xmin=0, xmax=0.03, color=colors[i], label=name, lw=0)
    for lo, hi in overlaps:
        ax.axhspan(lo, hi, color='tab:orange', alpha=0.12, lw=0)
    ax.plot(m_before.mean(axis=(1, 2)), y, color='tab:red', lw=1.2, label='before')
    ax.plot(m_after.mean(axis=(1, 2)), y, color='tab:blue', lw=1.2, label='after')
    ax.set_ylim(heights_extent)
    ax.set_xlabel('fraction of a layer that is sample')
    ax.set_ylabel('height (mm)')
    ax.set_title('Sample in each layer (shaded: overlaps)', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')

    shifts = after['merge_shifts_voxels']
    note = ', '.join(f'{n}: ({s[0]:+.2f}, {s[1]:+.2f})' for n, s in zip(names, shifts))
    fig.suptitle(f'Vertical cross sections of the merged mask (sections at voxel {section[0]} and {section[1]})\n'
                 f'registration shifts in voxels (axis 1, axis 2): {note}', fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    logger.info(f'wrote {path}')
    return path


def merge_mask_files(srcs, dst, offsets=None, ramsz=None, spec_log=None, tomo_scans=None, reference=None,
                     register=True, register_slices=5, max_shift=10.0, figure=None):
    """Merge the ``*_tomo_mask.npz`` files ``srcs`` into ``dst``, see the module's description

    The order of ``srcs`` is the order of ``offsets``, ``ramsz`` and ``tomo_scans``.  With ``figure``, the path of
    a png, it also draws the cross sections of ``plot_merge`` (only when the masks are registered, as it is the
    difference that it shows).  Returns what was written.
    """
    names = [os.path.basename(s) for s in srcs]
    masks = []
    for s in srcs:
        with np.load(s) as f:
            masks.append({k: f[k] for k in f.files})
    check_masks(masks)
    offsets, ref, ref_ramsz = resolve_offsets(masks, offsets, ramsz, spec_log, tomo_scans, reference, names)
    logger.info(f'Reference mask {names[ref]}, offsets {np.round(offsets, 4).tolist()} mm')
    out = merge_masks(masks, offsets, ref, register, register_slices, max_shift, names)
    if figure is not None:
        if register:
            before = merge_masks(masks, offsets, ref, False, names=names)
            plot_merge(before, out, [n.replace('_tomo_mask.npz', '') for n in names], figure)
        else:
            logger.info('Not drawing the cross sections, they show the difference that the registration makes')
    out['merged_from'] = np.array(json.dumps(names))
    if ref_ramsz is not None:
        out['ramsz'] = np.array(ref_ramsz)
    else:
        logger.warning('The ramsz of the reference scan is not known, so the merged mask does not record it. '
                       'The NF reconstruction then needs vertical_motor_position in its config')
    np.savez_compressed(dst, **out)
    logger.info(f'wrote {dst}')
    return out
