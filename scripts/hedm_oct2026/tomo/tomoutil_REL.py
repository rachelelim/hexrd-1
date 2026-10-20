from __future__ import annotations

from pathlib import Path
import logging
from typing import Sequence

import h5py
import hdf5plugin  # noqa: F401  registers the HDF5 LZ4 filter the Retiga files use
import numpy as np
import scipy.ndimage as ndi
from skimage.transform import iradon, resize

try:
    import imageio.v2 as imageio
except ImportError:
    from skimage import io as imageio


logger = logging.getLogger(__name__)


def _validate_2d_shape(name: str, arr: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if arr.shape != expected_shape:
        raise ValueError(
            f"{name} shape {arr.shape} does not match expected {expected_shape}")


def _build_image_paths(
    data_folder: str | Path,
    img_start: int,
    num_imgs: int,
    stem: str = "nf_",
    num_digits: int = 5,
    ext: str = ".tif",
) -> list[Path]:
    folder = Path(data_folder)
    return [
        folder / f"{stem}{img_num:0{num_digits}d}{ext}"
        for img_num in range(img_start, img_start + num_imgs)
    ]


def _load_tiff_stack(
    paths: Sequence[Path],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if not paths:
        raise ValueError("No image paths provided")

    first = np.asarray(imageio.imread(paths[0]), dtype=dtype)
    stack = np.empty((len(paths), *first.shape), dtype=dtype)
    stack[0] = first

    for i, path in enumerate(paths[1:], start=1):
        stack[i] = np.asarray(imageio.imread(path), dtype=dtype)

    return stack


def _load_h5_stack(
    filename: str | Path,
    dataset_path: str = "entry/data/data/",
    rows: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load an HDF5 image stack as float32.

    rows=(r0, r1) loads only detector rows r0:r1 of every frame, one frame at
    a time, so the full-frame stack never has to fit in memory.
    """
    with h5py.File(filename, "r") as h5f:
        dset = h5f[dataset_path]
        if rows is None:
            return np.asarray(dset, dtype=np.float32)
        r0, r1 = rows
        stack = np.empty((dset.shape[0], r1 - r0, dset.shape[2]), dtype=np.float32)
        for i in range(dset.shape[0]):
            stack[i] = dset[i, r0:r1, :]
        return stack


def h5_frame_sums(
    filename: str | Path,
    dataset_path: str = "entry/data/data/",
) -> np.ndarray:
    """Total counts of each full frame (read one frame at a time)."""
    with h5py.File(filename, "r") as h5f:
        dset = h5f[dataset_path]
        return np.array(
            [dset[i].sum(dtype=np.float64) for i in range(dset.shape[0])])


def read_frame_timestamps(
    filename: str | Path,
    attr_group: str = "entry/instrument/NDAttributes",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamp_s, unique_id) per frame from the areaDetector NDAttributes."""
    with h5py.File(filename, "r") as h5f:
        grp = h5f[attr_group]
        sec = grp["NDArrayEpicsTSSec"][()].astype(np.float64)
        nsec = grp["NDArrayEpicsTSnSec"][()].astype(np.float64)
        uid = grp["NDArrayUniqueId"][()]
    return sec + nsec * 1e-9, uid


def frame_slots_from_timestamps(
    timestamps: np.ndarray,
    period: float | None = None,
    tol: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Assign each retained frame its original trigger index ("slot").

    An interval of k*period between consecutive frames (within +-tol*period)
    means k-1 triggers were missed.  Returns (slot, missing_slots, period).
    """
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    if period is None:
        period = float(np.median(dt))
    k = np.round(dt / period)
    off = np.abs(dt / period - k)
    if np.any(off > tol) or np.any(k < 1):
        bad = np.flatnonzero((off > tol) | (k < 1))
        raise RuntimeError(
            "Frame intervals are not an integer number of trigger periods at "
            f"frames {bad.tolist()}: dt/period = {np.round(dt[bad] / period, 3).tolist()}")
    slot = np.concatenate([[0], np.cumsum(k.astype(int))])
    missing = np.setdiff1d(np.arange(slot[-1] + 1), slot)
    return slot, missing, period


def frame_slots_from_h5(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Trigger slot per frame and list of missing slots for an areaDetector HDF5 file."""
    t, uid = read_frame_timestamps(filename)
    slot, missing, period = frame_slots_from_timestamps(t)
    if not np.all(np.diff(uid) == 1):
        logger.warning("NDArrayUniqueId is not contiguous: frames were also lost "
                       "after the camera driver")
    logger.info("%s: %d frames, %d trigger slots, %d missing (period %.4f s)",
                Path(filename).name, len(slot), slot[-1] + 1, len(missing), period)
    if len(missing):
        logger.info("missing trigger slots: %s", ", ".join(map(str, missing.tolist())))
    return slot, missing


def gen_median_image_h5(
    filename: str | Path,
    dataset_path: str = "entry/data/data/",
    rows: tuple[int, int] | None = None,
) -> np.ndarray:
    logger.info("Loading HDF5 data for median image")
    stack = _load_h5_stack(filename, dataset_path, rows)
    logger.info("Computing median image")
    return np.median(stack, axis=0)


def gen_median_image(
    data_folder: str | Path,
    img_start: int,
    num_imgs: int,
    stem: str = "nf_",
    num_digits: int = 5,
    ext: str = ".tif",
) -> np.ndarray:
    logger.info("Loading TIFF data for median image")
    paths = _build_image_paths(
        data_folder, img_start, num_imgs, stem, num_digits, ext)
    stack = _load_tiff_stack(paths)
    logger.info("Computing median image")
    return np.median(stack, axis=0)


def _compute_attenuation(
    image_stack: np.ndarray,
    tbf: np.ndarray,
    tdf: np.ndarray | None = None,
    gaussian_sigma: float = 2.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if image_stack.ndim != 3:
        raise ValueError(
            f"image_stack must be 3D, got shape {image_stack.shape}")

    nimgs, nrows, ncols = image_stack.shape
    expected_shape = (nrows, ncols)

    tbf = np.asarray(tbf, dtype=np.float32)
    _validate_2d_shape("tbf", tbf, expected_shape)

    if tdf is None:
        tdf = np.zeros(expected_shape, dtype=np.float32)
    else:
        tdf = np.asarray(tdf, dtype=np.float32)
        _validate_2d_shape("tdf", tdf, expected_shape)

    denominator = np.clip(tbf - tdf, eps, None)

    # One frame at a time: same arithmetic as doing the whole stack at once,
    # without holding several full-stack temporaries.
    rad_stack = np.empty(image_stack.shape, dtype=np.float32)
    min_rad = np.inf
    any_invalid = False
    for i in range(nimgs):
        ratio = np.clip(
            (image_stack[i].astype(np.float32) - tdf) / denominator, eps, None)
        rad = -np.log(ratio)
        bad = ~np.isfinite(rad)
        any_invalid |= bool(bad.any())
        finite = rad[~bad]
        if finite.size:
            min_rad = min(min_rad, finite.min())
        rad_stack[i] = rad

    if not np.isfinite(min_rad):
        raise ValueError("All attenuation values are invalid")
    if any_invalid:
        rad_stack[~np.isfinite(rad_stack)] = min_rad

    filtered_rad_stack = np.empty_like(rad_stack)
    for i in range(nimgs):
        filtered_rad_stack[i] = ndi.gaussian_filter(
            rad_stack[i], sigma=gaussian_sigma)

    return rad_stack, filtered_rad_stack


def gen_attenuation_rads_h5(
    filename: str | Path,
    tbf: np.ndarray,
    tdf: np.ndarray | None = None,
    dataset_path: str = "entry/data/data/",
    rows: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_stack = _load_h5_stack(filename, dataset_path, rows)
    rad_stack, filtered_rad_stack = _compute_attenuation(image_stack, tbf, tdf)
    return rad_stack, image_stack, filtered_rad_stack


def gen_filtered_rads_h5(
    filename: str | Path,
    tbf: np.ndarray,
    tdf: np.ndarray | None = None,
    dataset_path: str = "entry/data/data/",
    rows: tuple[int, int] | None = None,
    gaussian_sigma: float = 2.0,
    eps: float = 1e-8,
    out_dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming gen_attenuation_rads_h5: one frame in memory at a time.

    Returns (filtered_rad_stack, frame_sums). The filtered stack is stored
    directly as out_dtype (float16 by default, which is what the
    reconstruction uses), so peak memory is the output stack alone. Each frame
    gets the same arithmetic as _compute_attenuation. frame_sums are the
    total counts of each *full* frame (not just the rows band), for judging
    bad frames.
    """
    with h5py.File(filename, "r") as h5f:
        dset = h5f[dataset_path]
        nimgs, nrows, ncols = dset.shape
        r0, r1 = (0, nrows) if rows is None else rows
        expected_shape = (r1 - r0, ncols)

        tbf = np.asarray(tbf, dtype=np.float32)
        _validate_2d_shape("tbf", tbf, expected_shape)
        if tdf is None:
            tdf = np.zeros(expected_shape, dtype=np.float32)
        else:
            tdf = np.asarray(tdf, dtype=np.float32)
            _validate_2d_shape("tdf", tdf, expected_shape)
        denominator = np.clip(tbf - tdf, eps, None)

        filtered = np.empty((nimgs, *expected_shape), dtype=out_dtype)
        frame_sums = np.empty(nimgs, dtype=np.float64)
        for i in range(nimgs):
            frame = dset[i]
            frame_sums[i] = frame.sum(dtype=np.float64)
            ratio = np.clip(
                (frame[r0:r1].astype(np.float32) - tdf) / denominator, eps, None)
            rad = -np.log(ratio)
            if not np.isfinite(rad).all():
                # The non-streaming path would replace these with the stack
                # minimum, which is unknown until the whole stack is read.
                raise ValueError(
                    f"non-finite attenuation in frame {i}; use "
                    "gen_attenuation_rads_h5 for this stack")
            filtered[i] = ndi.gaussian_filter(rad, sigma=gaussian_sigma)
            if i % 100 == 0:
                logger.info(f"filtered frame {i}/{nimgs}")
    return filtered, frame_sums


def gen_attenuation_rads(
    tomo_data_folder: str | Path,
    tbf: np.ndarray,
    tomo_img_start: int,
    tomo_num_imgs: int,
    stem: str = "nf_",
    num_digits: int = 5,
    ext: str = ".tif",
    tdf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _build_image_paths(
        tomo_data_folder,
        tomo_img_start,
        tomo_num_imgs,
        stem,
        num_digits,
        ext,
    )
    image_stack = _load_tiff_stack(paths)
    rad_stack, filtered_rad_stack = _compute_attenuation(image_stack, tbf, tdf)
    return rad_stack, image_stack, filtered_rad_stack


def tomo_reconstruct_layer(
    rad_stack: np.ndarray,
    cross_sectional_dim: float = 1.5,
    layer_row: int = 1024,
    start_tomo_ang: float = 0.0,
    end_tomo_ang: float = 360.0,
    y_center: float = 0.0,
    pixel_size: float = 0.00148,
    bad_frames: Sequence[int] | None = None,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct one layer by filtered back-projection.

    If `theta` (degrees, one per frame in rad_stack) is given it is used as is;
    this is required when frames were dropped during acquisition.  Otherwise
    the angles are spread uniformly over [start_tomo_ang, end_tomo_ang).
    """
    if rad_stack.ndim != 3:
        raise ValueError(f"rad_stack must be 3D, got shape {rad_stack.shape}")

    num_imgs = rad_stack.shape[0]
    if theta is None:
        theta = np.linspace(start_tomo_ang, end_tomo_ang, num_imgs, endpoint=False)
    else:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (num_imgs,):
            raise ValueError(
                f"theta has {theta.size} entries but rad_stack has {num_imgs} frames")

    if bad_frames is not None and len(bad_frames) > 0:
        rad_stack = np.delete(rad_stack, bad_frames, axis=0)
        theta = np.delete(theta, bad_frames)

    if not (0 <= layer_row < rad_stack.shape[1]):
        raise IndexError(
            f"layer_row {layer_row} out of bounds for shape {rad_stack.shape}")

    sinogram = rad_stack[:, layer_row, :]

    rotation_axis_pos = -int(np.round(y_center / pixel_size))
    max_rad = int(cross_sectional_dim / pixel_size / 2.0 * 1.1)

    if rotation_axis_pos >= 0:
        sinogram_cut = sinogram[:, 2 * rotation_axis_pos:]
    else:
        sinogram_cut = sinogram[:, : 2 * rotation_axis_pos]

    half_width = int(np.round(sinogram_cut.shape[1] / 2.0))
    dist_from_edge = half_width - max_rad
    if dist_from_edge > 0:
        sinogram_cut = sinogram_cut[:, dist_from_edge:-dist_from_edge]

    reconstruction = iradon(sinogram_cut.T, theta=theta, circle=True)
    return np.rot90(reconstruction, 3)


def threshold_and_clean_tomo_layer(
    reconstruction_fbp: np.ndarray,
    recon_thresh: float,
    noise_obj_size: int,
    min_hole_size: int,
    edge_cleaning_iter: int | None = None,
    erosion_iter: int = 1,
    dilation_iter: int = 4,
    powder_open_um: float = 0.0,
    pixel_size_um: float | None = None,
    powder_restore_um: float = 0.0,
    powder_min_object_um2: float | None = None,
    powder_fill_holes_um2: float | None = None,
) -> np.ndarray:
    """Threshold the FBP and clean it at full resolution.

    Powder removal (powder_open_um > 0, needs pixel_size_um): after the
    closing, a morphological opening with a disk of radius powder_open_um
    deletes everything narrower than twice that radius (powder particles,
    necks and thin wall roughness) while keeping the thick lattice walls.
    powder_restore_um then gives back original pixels within that distance of
    the opened shape, so the true wall edge survives and only powder farther
    out stays removed. powder_min_object_um2 / powder_fill_holes_um2, if given,
    replace noise_obj_size / min_hole_size with sizes in um^2.
    """
    binary = reconstruction_fbp > recon_thresh

    if dilation_iter > 0:
        binary = ndi.binary_dilation(binary, iterations=dilation_iter)
    if erosion_iter > 0:
        binary = ndi.binary_erosion(binary, iterations=erosion_iter)

    if powder_open_um > 0:
        if not pixel_size_um or pixel_size_um <= 0:
            raise ValueError("pixel_size_um is required for powder removal")
        r_px = powder_open_um / pixel_size_um
        # opening = erode by a disk, then dilate back; both via distance maps
        core = ndi.distance_transform_edt(binary) > r_px
        opened = ndi.distance_transform_edt(~core) <= r_px
        if powder_restore_um > 0:
            near = (ndi.distance_transform_edt(~opened)
                    <= powder_restore_um / pixel_size_um)
            binary = binary & near
        else:
            binary = binary & opened
        px_area = pixel_size_um ** 2
        if powder_min_object_um2 is not None:
            noise_obj_size = powder_min_object_um2 / px_area
        if powder_fill_holes_um2 is not None:
            min_hole_size = powder_fill_holes_um2 / px_area

    labeled, num_labels = ndi.label(binary)
    if num_labels > 0:
        counts = np.bincount(labeled.ravel())
        remove_mask = counts < noise_obj_size
        remove_mask[0] = False
        binary[remove_mask[labeled]] = False

    holes, num_holes = ndi.label(~binary)
    if num_holes > 0:
        hole_counts = np.bincount(holes.ravel())
        fill_mask = (hole_counts >= 1) & (hole_counts < min_hole_size)
        fill_mask[0] = False
        binary[fill_mask[holes]] = True

    if edge_cleaning_iter is not None and edge_cleaning_iter > 0:
        binary = ndi.binary_erosion(binary, iterations=edge_cleaning_iter)
        binary = ndi.binary_dilation(binary, iterations=edge_cleaning_iter)

    return binary.astype(bool)


def crop_and_rebin_tomo_layer(
    binary_recon: np.ndarray,
    voxel_spacing: float,
    pixel_size: float,
    x_center: float = 0.0,
    z_center: float = 0.0,
    x_length: float = 1.5,
    z_length: float = 1.5,
    circular_mask_rad: float | None = None,
) -> np.ndarray:
    if voxel_spacing <= 0 or pixel_size <= 0:
        raise ValueError("voxel_spacing and pixel_size must be positive")

    scaling = voxel_spacing / pixel_size
    rows, cols = binary_recon.shape

    new_rows = max(1, int(round(rows / scaling)))
    new_cols = max(1, int(round(cols / scaling)))

    rebinned = resize(
        binary_recon.astype(np.uint8),
        (new_rows, new_cols),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)

    row_extent = rebinned.shape[0] * voxel_spacing
    col_extent = rebinned.shape[1] * voxel_spacing

    cut_edge_x = int(round((row_extent - x_length) /
                     (2.0 * voxel_spacing) + x_center / voxel_spacing))
    cut_edge_z = int(round((col_extent - z_length) /
                     (2.0 * voxel_spacing) + z_center / voxel_spacing))

    x0, x1 = cut_edge_x, cut_edge_x + int(round(x_length / voxel_spacing))
    z0, z1 = cut_edge_z, cut_edge_z + int(round(z_length / voxel_spacing))

    x0 = max(0, x0)
    z0 = max(0, z0)
    x1 = min(rebinned.shape[0], x1)
    z1 = min(rebinned.shape[1], z1)

    cropped = rebinned[x0:x1, z0:z1]

    if circular_mask_rad is not None and cropped.size > 0:
        center_x = (cropped.shape[0] - 1) / 2.0
        center_z = (cropped.shape[1] - 1) / 2.0
        radius = circular_mask_rad / voxel_spacing

        yy, xx = np.ogrid[:cropped.shape[0], :cropped.shape[1]]
        mask = (yy - center_x) ** 2 + (xx - center_z) ** 2 > radius ** 2
        cropped = cropped.copy()
        cropped[mask] = False

    return cropped
