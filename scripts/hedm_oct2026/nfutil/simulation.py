"""Fast simulation of the diffraction peaks of an orientation.

The refinement and the orientation search simulate the peaks of hundreds of
thousands of orientations, and almost all of that time is spent in
``hexrd.hedm.xrdutil.simulateGVecs``. Most of what it does is the same for every
orientation or is never used here: it rebuilds the list of symmetry-equivalent
reflections on every call, and it computes angular pixel sizes that are thrown
away. ``simulate_angles`` does the same calculation for the angles with the
reflection list cached, and returns exactly the angles ``simulateGVecs`` does.

One step is skipped only when it cannot matter: ``simulateGVecs`` drops peaks
whose projection falls outside ``panel_dims``. If the geometry guarantees every
peak lands inside, the projection is skipped. Otherwise it is done as before.

contributing authors: lim37
"""

import numpy as np

from hexrd import constants
from hexrd.core.transforms import xfcapi
from hexrd.hedm.xrdutil.utils import _project_on_detector_plane

# The panel the original code simulates onto (mm), see _precompute_diffraction_data_*
PANEL_DIMS = ((-10, -10), (10, 10))
ETA_RANGE = ((-np.pi, np.pi),)
V_INV = np.array([1., 1., 1., 0., 0., 0.])    # unstrained
BEAM_VEC = np.ascontiguousarray(constants.beam_vec)

_TABLES = {}


class _PeakTable:
    """What simulate_angles needs that does not depend on the orientation."""

    def __init__(self, plane_data, detector_params, panel_dims):
        self.hkls = np.hstack(plane_data.getSymHKLs(withID=True)).T
        self.hkls_xyz = np.ascontiguousarray(self.hkls[:, 1:])
        self.b_mat = plane_data.latVecOps['B']
        self.wavelength = plane_data.wavelength
        self.chi = detector_params[6]
        self.t_vec_d = np.ascontiguousarray(detector_params[3:6])
        self.t_vec_s = np.ascontiguousarray(detector_params[7:10])
        self.r_mat_d = xfcapi.make_detector_rmat(detector_params[:3])
        self.panel_dims = panel_dims
        self.needs_panel_filter = not self._panel_cannot_cut(plane_data, detector_params)

    def _panel_cannot_cut(self, plane_data, detector_params):
        """True if no peak can project outside the panel.

        A peak scattered through at most tThMax lands within about D*tan(tThMax
        + tilt) of where the beam hits the detector, D being the distance from
        the sample, plus the offset of the detector origin and the sample. If
        that is inside the panel for any orientation, the filter is not needed.
        """
        two_theta_max = plane_data.tThMax
        if two_theta_max is None:
            return False
        tilt = np.arccos(np.clip((np.trace(self.r_mat_d) - 1.)/2., -1., 1.))
        if two_theta_max + tilt >= 1.2:
            return False
        distance = np.linalg.norm(self.t_vec_d) + np.linalg.norm(self.t_vec_s)
        reach = (distance*np.tan(two_theta_max + tilt) + np.hypot(*self.t_vec_d[:2])
                 + np.linalg.norm(self.t_vec_s))
        half_panel = min(abs(v) for corner in self.panel_dims for v in corner)
        return reach < half_panel


def _table(plane_data, detector_params, panel_dims):
    key = (plane_data.tThMax, plane_data.wavelength, plane_data.exclusions.tobytes(),
           plane_data.latVecOps['B'].tobytes(), np.asarray(detector_params).tobytes(),
           tuple(tuple(c) for c in panel_dims))
    table = _TABLES.get(key)
    if table is None:
        if len(_TABLES) > 64:
            _TABLES.clear()
        table = _TABLES[key] = _PeakTable(plane_data, detector_params, panel_dims)
    return table


def simulate_angles(experiment, exp_map, r_mat_c=None, panel_dims=PANEL_DIMS):
    """The (tth, eta, omega) of every peak of an unstrained orientation, (n, 3).

    Equal to ``simulateGVecs(...)[2]`` with the arguments the nfutil
    precompute functions use (experiment.plane_data, detector_params, ome_range
    and ome_period, an unstrained grain at the sample origin).
    """
    plane_data = experiment.plane_data
    table = _table(plane_data, experiment.detector_params, panel_dims)
    if r_mat_c is None:
        r_mat_c = xfcapi.make_rmat_of_expmap(np.asarray(exp_map).reshape(3))
    angles = np.vstack(xfcapi.oscill_angles_of_hkls(
        table.hkls_xyz, table.chi, r_mat_c, table.b_mat, table.wavelength,
        v_inv=V_INV, beam_vec=BEAM_VEC))

    # keep the peaks inside the eta and omega ranges, as _filter_hkls_eta_ome does
    mask_eta = np.zeros(len(angles), dtype=bool)
    for etas in ETA_RANGE:
        mask_eta = np.logical_or(mask_eta, xfcapi.validate_angle_ranges(angles[:, 1], etas[0], etas[1]))
    ccw = True
    mask_ome = np.zeros(len(angles), dtype=bool)
    for omes in experiment.ome_range:
        if omes[1] - omes[0] < 0:
            ccw = False
        mask_ome = np.logical_or(mask_ome, xfcapi.validate_angle_ranges(angles[:, 2], omes[0], omes[1], ccw=ccw))
    mask = np.logical_and(mask_eta, mask_ome)
    angles = angles[mask, :]

    if len(angles) and not table.needs_panel_filter:
        # simulateGVecs projects the peaks onto the panel to drop those outside it. The
        # projection drops rows with NaN angles (hkls with no solution), and simulateGVecs
        # then indexes the full list with positions from the shorter one, so when there
        # are NaN rows it keeps the first n_finite rows, which are not the finite ones.
        # The angles are reproduced as they are, so scores and references do not change.
        n_finite = int(np.isfinite(angles).all(axis=1).sum())
        if n_finite < len(angles):
            angles = angles[:n_finite, :]
    elif len(angles):
        det_xy, _, _ = _project_on_detector_plane(
            angles, table.r_mat_d, r_mat_c, table.chi, table.t_vec_d,
            np.zeros(3), table.t_vec_s, None, beamVec=BEAM_VEC)
        lo, hi = table.panel_dims
        on_panel = np.logical_and(
            np.logical_and(det_xy[:, 0] >= lo[0], det_xy[:, 0] <= hi[0]),
            np.logical_and(det_xy[:, 1] >= lo[1], det_xy[:, 1] <= hi[1]))
        angles = angles[np.where(on_panel)[0], :]

    # map omega into the period, as hexrd's mapAngle does
    period = np.asarray(experiment.ome_period, dtype=np.float64)
    min_val, max_val = period.min(), period.max()
    omega = np.nan_to_num(angles[:, 2])
    val = np.mod(omega - min_val, max_val - min_val) + min_val
    val[np.logical_and(val == min_val, omega > min_val)] = max_val
    angles[:, 2] = val
    return angles
