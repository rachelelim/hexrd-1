"""Per-voxel processors, multi-processor handlers and distributors.

contributing authors: dcp5303, ken38, seg246, Austin Gerlt, Simon Mason, lim37
"""

import contextlib
import h5py
import logging
import multiprocessing
import numba
import numpy as np
import os
import scipy
import shutil
import skimage
import tempfile
import timeit
from hexrd import rotations
from hexrd import xrdutil
from hexrd.transforms import xfcapi
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

from .data_io import _load_images

def make_chunks(n_items, ncpus, chunk_size=-1):
    """Split ``n_items`` into contiguous chunks for the multiprocessing pool.

    If ``chunk_size`` is -1 it is chosen so there is about one chunk per CPU.
    Returns ``chunks, starts, stops, chunk_size`` where chunk ``i`` covers
    ``[starts[i], stops[i])`` and the last stop is clipped to ``n_items``.
    """
    if chunk_size == -1:
        chunk_size = int(np.ceil(n_items/ncpus))
    num_chunks = int(np.ceil(n_items/chunk_size))
    chunks = np.arange(num_chunks)
    starts = chunks*chunk_size
    stops = np.minimum(starts + chunk_size, n_items)
    return chunks, starts, stops, chunk_size


def worker_init(id_state, id_exp):
    """process initialization function. This function is only used when the
    child processes are spawned (instead of forked). When using the fork model
    of multiprocessing the data is just inherited in process memory."""
    import joblib

    global _mp_state
    state = joblib.load(id_state)
    experiment = joblib.load(id_exp)
    _mp_state = state + (experiment,)


def set_multiprocessing_method(multiprocessing_start_method):
    # Set multiprocessing method if not already done.
    # allow_none=True so that asking does not fix the default method (which is
    # 'forkserver' on Linux from Python 3.14), and force=True because the method
    # may already have been fixed, by an earlier call or by the notebook
    if multiprocessing.get_start_method(allow_none=True) != multiprocessing_start_method:
        multiprocessing.set_start_method(
            multiprocessing_start_method, force=True)


@contextlib.contextmanager
def multiprocessing_pool(ncpus, state):
    """function that handles the initialization of multiprocessing. It handles
    properly the use of spawned vs forked multiprocessing. The multiprocessing
    can be either 'fork' or 'spawn', with 'spawn' being required in non-fork
    platforms (like Windows) and 'fork' being preferred on fork platforms due
    to its efficiency.
    """
    # state = ( chunk_size,
    #           image_stack,
    #           angles,
    #           precomp,
    #           coords,
    #           experiment )

    if multiprocessing.get_start_method() == 'fork':
        # Use FORK multiprocessing.

        # All read-only data can be inherited in the process. So we "pass" it
        # as a global that the child process will be able to see. At the end of
        # theprocessing the global is removed.
        global _mp_state
        _mp_state = state
        pool = multiprocessing.Pool(ncpus)
        yield pool
        del (_mp_state)
    else:
        # Use SPAWN multiprocessing.

        # As we can not inherit process data, all the required data is
        # serialized into a temporary directory using joblib. The
        # multiprocessing pool will have the "worker_init" as initialization
        # function that takes the key for the serialized data, which will be
        # used to load the parameter memory into the spawn process (also using
        # joblib). In theory, joblib uses memmap for arrays if they are not
        # compressed, so no compression is used for the bigger arrays.
        import joblib
        tmp_dir = tempfile.mkdtemp(suffix='-nf-grand-loop')
        try:
            # dumb dumping doesn't seem to work very well.. do something ad-hoc
            logging.info('Using "%s" as temporary directory.', tmp_dir)

            id_exp = joblib.dump(state[-1],
                                 os.path.join(tmp_dir,
                                              'grand-loop-experiment.gz'),
                                 compress=True)
            id_state = joblib.dump(state[:-1],
                                   os.path.join(tmp_dir, 'grand-loop-data'))
            pool = multiprocessing.Pool(ncpus, worker_init,
                                        (id_state[0], id_exp[0]))
            yield pool
        finally:
            logging.info('Deleting "%s".', tmp_dir)
            shutil.rmtree(tmp_dir)


@numba.njit(nogil=True, cache=True)
def _quant_and_clip_confidence(coords, angles, image,
                               base, inv_deltas, clip_vals, bsp, ome_edges):
    """quantize and clip the parametric coordinates in coords + angles

    coords - (..., 2) array: input 2d parametric coordinates
    angles - (...) array: additional dimension for coordinates
    base   - (3,) array: base value for quantization (for each dimension)
    inv_deltas - (3,) array: inverse of the quantum size (for each dimension)
    clip_vals - (2,) array: clip size (only applied to coords dimensions)
    bsp - (2,) array: beam stop vertical position and width in mm
    ome_edges - (...): list of omega edges

    clipping is performed on ranges [0, clip_vals[0]] for x and
    [0, clip_vals[1]] for y

    returns an array with the quantized coordinates, with coordinates
    falling outside the clip zone filtered out.

    """
    # Added binary serach function with Zack Singer's help - SEG 06/23/2023
    def find_target_index(array, target):
        global_index = 0
        while len(array) > 1:
            index = len(array) // 2
            if array[index] == target:
                return global_index + index
            elif array[index] > target:
                array = array[:index]
            else:
                global_index = global_index + index
                array = array[index:]
        return global_index

    # Main function
    count = len(coords)

    in_sensor = 0
    matches = 0
    for i in range(count):
        xf = coords[i, 0]
        yf = coords[i, 1]

        # does not count intensity which is covered by the beamstop dcp 5.13.21
        # if len(bsp) == 2: # Added this flag for handling if we have a mask type beamstop - SEG 10/28/2023
        #     if abs(yf-bsp[0]) < (bsp[1]/2.):
        #         continue

        xf = np.floor((xf - base[0]) * inv_deltas[0])
        if not xf >= 0.0:
            continue
        if not xf < clip_vals[0]:
            continue

        yf = np.floor((yf - base[1]) * inv_deltas[1])

        if not yf >= 0.0:
            continue
        if not yf < clip_vals[1]:
            continue

        # Adding 2D 'beamstop' mask functionality to handle the 2x objective lens + scinitaltor issues - SEG 10/28/2023
        # The beamstop parameter is now the shape of a single image
        # The beamstop mask is TRUE on the beamstop/past the edge of scintilator
        # Comment out the top bsp function
        # if len(bsp) > 2:
        if bsp[int(yf), int(xf)]:
            continue

        # CHANGE SEG 6/22/2023 and 10/03/2023 - Put in a binary serach of the omega edges
        ome_pos = angles[i]
        # While bisect left is nominally faster - it does not work with numba
        # Bisect left returns the index, j, such that all ome_edges[0:j] < ome_pos
        # zf = bisect.bisect_left(ome_edges, ome_pos) - 1
        # This method is faster when combined with numba
        zf = find_target_index(ome_edges, ome_pos)
        in_sensor += 1

        x, y, z = int(xf), int(yf), int(zf)

        if image[z, y, x]:
            matches += 1

    return 0 if in_sensor == 0 else float(matches)/float(in_sensor)


# Search radii (degrees) used when refining an orientation at a voxel
REFINEMENT_RADII_DEG = [5.0, 4., 3.5, 3., 2.5, 2., 1.5, 1.0, 0.5, 0.25, 0.1]


def _pattern_search_refine(start_exp_map, cost, radii_deg, max_recenters=100):
    """Coarse-to-fine pattern search for the orientation with the highest cost.

    At each radius (in degrees) a 3x3x3 grid of exp-map offsets (-r, 0, +r on
    each axis) is evaluated around the current best orientation. The grid is
    then re-centered on the best point and evaluated again, as long as that
    strictly improves the cost (at most ``max_recenters`` times), before moving
    to the next, smaller radius. A pass that does not improve on the previous
    one is discarded.

    ``cost`` takes an (n, 3) array of exp-maps and returns n confidences.
    Returns ``exp_map, confidence`` of the best orientation found.
    """
    exp_map = np.asarray(start_exp_map, dtype=float)
    confidence = 0.0
    for mis in radii_deg:
        mis_amt = np.radians(mis)
        # Create a linup of the orientations to go on either side
        ori_pts = np.array([-mis_amt, 0., mis_amt])
        XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts)  # Make that 3D
        grid0 = np.vstack(
            [XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T  # Re-arange
        for counter in range(max_recenters):
            all_confidence = cost(grid0 + exp_map)
            # Grab just the first instance if there is a tie
            idx = np.argmax(all_confidence)
            # The first pass at each radius is always taken. Later passes
            # (re-centering) are only taken while they strictly improve.
            if counter > 0 and all_confidence[idx] <= confidence:
                break
            exp_map = grid0[idx] + exp_map
            confidence = all_confidence[idx]
    return exp_map, confidence


def _test_single_orientation_at_single_coordinate(experiment, image_stack, coord_to_test, orientation_data_to_test, refine_yes_no=0):
    """
        Goal:

        Input:

        Output:

    """
    if refine_yes_no == 0:
        # No refinement needed - just test the orientation
        # Unpack the precomputed orientation data
        exp_map, angles, rMat_ss, gvec_cs, rMat_c = orientation_data_to_test
        # Grab some experiment data
        tD = experiment.tVec_d  # Detector X,Y,Z translation (mm)
        rD = experiment.rMat_d  # Detector rotation matrix (rad)
        tS = experiment.tVec_s  # Sample X,Y,Z translation (mm)
        # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
        base = experiment.base
        # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
        inv_deltas = experiment.inv_deltas
        clip_vals = experiment.clip_vals  # Number of pixels along X,Y [mm,mm]
        # Beam stop parameters [vertical center,width] [mm,mm]
        bsp = experiment.bsp
        # Omega start stop positions for each frame in image stack
        ome_edges = experiment.ome_edges
        # Find where those g-vectors intercept the detector from our coordinate point
        det_xy = xfcapi.gvec_to_xy(
            gvec_cs, rD, rMat_ss, np.squeeze(rMat_c), tD, tS, coord_to_test)
        # Check xy detector positions and omega value to see if intensity exisits
        confidence = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                                base, inv_deltas, clip_vals, bsp, ome_edges)
        # Return the orienation and its confidence
        misorientation = 0
    elif refine_yes_no == 1:
        # Refinement needed
        # Unpack the precomputed orientation data
        original_exp_map = orientation_data_to_test[0]
        very_original_exp_map = orientation_data_to_test[0]
        # Grab some experiment data
        # Packaged information about the material and HKLs
        plane_data = experiment.plane_data
        # Detector tilts, position, as well as stage position and chi [?,mm,mm,chi]
        detector_params = experiment.detector_params
        pixel_size = experiment.pixel_size  # Pixel size (mm)
        # Start and stop omega position of image stack (rad)
        ome_range = experiment.ome_range
        # Defined omega period for HEXRD to work in (rad)
        ome_period = experiment.ome_period
        tD = experiment.tVec_d  # Detector X,Y,Z translation (mm)
        rD = experiment.rMat_d  # Detector rotation matrix (rad)
        tS = experiment.tVec_s  # Sample X,Y,Z translation (mm)
        # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
        base = experiment.base
        # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
        inv_deltas = experiment.inv_deltas
        clip_vals = experiment.clip_vals  # Number of pixels along X,Y [mm,mm]
        # Beam stop parameters [vertical center,width] [mm,mm]
        bsp = experiment.bsp
        # Omega start stop positions for each frame in image stack
        ome_edges = experiment.ome_edges
        # Pixels near the edge of the detector to avoid
        panel_dims_expanded = [(-10, -10), (10, 10)]
        # Assume grain is unstrained
        ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])

        def cost(trial_exp_maps):
            # Confidence of each trial orientation at the coordinate point
            confidences = np.zeros(trial_exp_maps.shape[0])
            for i, exp_map in enumerate(trial_exp_maps):
                # Transform exp_map to rotation matrix
                rMat_c = xfcapi.make_rmat_of_expmap(exp_map)
                # Define all parameters for the orientation (strain and orientation)
                gparams = np.hstack([exp_map, ref_gparams])
                # Simulate the the diffraction events
                sim_results = xrdutil.simulateGVecs(plane_data, detector_params, gparams, panel_dims=panel_dims_expanded,
                                                    pixel_pitch=pixel_size, ome_range=ome_range, ome_period=ome_period,
                                                    distortion=None)
                # Pull just the angles for each g-vector
                angles = sim_results[2]
                # Calculate the sample rotation matrix
                rMat_ss = xfcapi.make_sample_rmat(
                    experiment.chi, angles[:, 2])
                # Convert the angles to g-vectors
                gvec_cs = xfcapi.angles_to_gvec(
                    angles, chi=experiment.chi, rmat_c=rMat_c)
                # Find where those g-vectors intercept the detector from our coordinate point
                det_xy = xfcapi.gvec_to_xy(
                    gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test)
                # Check xy detector positions and omega value to see if intensity exisits
                confidences[i] = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                                            base, inv_deltas, clip_vals, bsp, ome_edges)
            return confidences

        exp_map, confidence = _pattern_search_refine(
            original_exp_map, cost, REFINEMENT_RADII_DEG)

        # What is the misorientation, in degrees between this exp_map and the original?
        original_quats = np.atleast_2d(
            rotations.quatOfExpMap(very_original_exp_map))
        refined_quats = np.atleast_2d(rotations.quatOfExpMap(exp_map))
        [misorientation, a] = rotations.misorientation(
            original_quats.T, refined_quats.T)  # In radians
        misorientation = np.degrees(misorientation)

    # Ensure output is the correct size
    if len(np.shape(exp_map)) == 1:
        exp_map = np.expand_dims(exp_map, 0)
    if len(np.shape(confidence)) == 0:
        confidence = np.expand_dims(confidence, 0)

    return exp_map, confidence, misorientation


def _test_single_orientation_at_many_coordinates(experiment, image_stack, coords_to_test, orientation_data_to_test):
    """
        Goal:
            Test a single orientation at a large number of coordinate points to check the
                confidence the orientation exists at each coordinate point.
        Input:

        Output:
            - Numpy array of size [# of coordinate points] containing a confidence value
                for each point.
    """

    # Grab some experiment data
    tD = experiment.tVec_d  # Detector X,Y,Z translation (mm)
    rD = experiment.rMat_d  # Detector rotation matrix (rad)
    tS = experiment.tVec_s  # Sample X,Y,Z translation (mm)
    # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
    base = experiment.base
    # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals  # Number of pixels along X,Y [mm,mm]
    # Beam stop parameters [vertical center,width] [mm,mm]
    bsp = experiment.bsp
    # Omega start stop positions for each frame in image stack
    ome_edges = experiment.ome_edges

    # Grab orientation information
    exp_map, angles, rMat_ss, gvec_cs, rMat_c = orientation_data_to_test

    # How many coordinate points do we have to test?
    n_coords = np.shape(coords_to_test)[0]

    # Initialize the confidence array to be returned
    all_confidence = np.zeros(n_coords)
    all_exp_maps = np.zeros([n_coords, 3])

    # Check each orientation at the coordinate point
    for i in np.arange(n_coords):
        # What is our coordinate?
        coord_to_test = coords_to_test[i, :]
        # Find intercept point of each g-vector on the detector
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, np.squeeze(
            rMat_c), tD, tS, coord_to_test)  # Convert angles to xy detector positions
        # Check detector positions and omega values to see if intensity exisits
        all_confidence[i] = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                                       base, inv_deltas, clip_vals, bsp, ome_edges)
        all_exp_maps[i, :] = exp_map

    # Return the confidence value at each coordinate point
    return all_exp_maps, all_confidence


def _test_many_orientations_at_single_coordinate(experiment, image_stack, coord_to_test, orientation_data_to_test):
    """
        Goal:
            Test many orientations against a single coordinate to determine which orientation is the best fit.  If desired
                this function can call the refinement funciton such that it produces the best orientation.
        Input:

        Output:
    """

    # Grab some experiment data
    tD = experiment.tVec_d  # Detector X,Y,Z translation (mm)
    rD = experiment.rMat_d  # Detector rotation matrix (rad)
    tS = experiment.tVec_s  # Sample X,Y,Z translation (mm)
    # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
    base = experiment.base
    # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals  # Number of pixels along X,Y [mm,mm]
    # Beam stop parameters [vertical center,width] [mm,mm]
    bsp = experiment.bsp
    # Omega start stop positions for each frame in image stack
    ome_edges = experiment.ome_edges

    # Unpack the orientation information
    all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c = orientation_data_to_test

    # How many orientations do we have?
    n_oris = np.shape(all_exp_maps)[0]

    # Initialize the confidence array to be returned
    all_confidence = np.zeros(n_oris)

    # Check each orientation at the coordinate point
    for i in np.arange(n_oris):
        # Grab orientation information
        angles = all_angles[i]
        rMat_ss = all_rMat_ss[i]
        gvec_cs = all_gvec_cs[i]
        rMat_c = all_rMat_c[i]
        # Find intercept point of each g-vector on the detector
        # Convert angles to xy detector positions
        det_xy = xfcapi.gvec_to_xy(
            gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test)
        # Check detector positions and omega values to see if intensity exisits
        all_confidence[i] = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                                       base, inv_deltas, clip_vals, bsp, ome_edges)

    # Find the index of the max confidence
    idx = np.where(all_confidence == np.max(all_confidence))[
        0][0]  # Grab just the first instance if there is a tie

    # What is the hightest confidence orientation and what is its confidence
    exp_map = all_exp_maps[idx]
    confidence = all_confidence[idx]

    return exp_map, confidence, idx


def _test_many_orientations_at_many_coordinates(experiment, image_stack, coordinates_to_test, orientation_data_to_test, refine_yes_no=0, start=0, stop=0):
    """
        Goal:
            This is a multiprocessing splitter
        Input:

        Output:
    """
    # Check which mode we are in.
    # If there are more coords than oris - we have chunked coords - run test_many_orientations_at_single_coordinate
    # If there are more oris than coords - we have chunked oris - run test_single_orientation_at_many_coordinates
    # How many orientations?
    n_oris = np.shape(orientation_data_to_test[0])[0]
    # How many coordinate points
    n_coords = np.shape(coordinates_to_test)[0]
    if n_coords >= n_oris:
        # We have chunked the coordinates
        coordinates_to_test = coordinates_to_test[start:stop]
        n_coords = np.shape(coordinates_to_test)[0]
        # Initalize arrays
        all_exp_maps = np.zeros([n_coords, 3])
        all_confidence = np.zeros(n_coords)
        all_idx = np.zeros(n_coords, int)
        all_misorientation = np.zeros(n_coords)
        # Loop over the coordinates
        for i in np.arange(n_coords):
            coord_to_test = coordinates_to_test[i, :]
            if n_oris == 1:
                exp_map, confidence, dummy = _test_single_orientation_at_single_coordinate(
                    experiment, image_stack, coord_to_test, orientation_data_to_test)
                idx = 0
            else:
                exp_map, confidence, idx = _test_many_orientations_at_single_coordinate(
                    experiment, image_stack, coord_to_test, orientation_data_to_test)
            all_exp_maps[i, :] = exp_map
            all_confidence[i] = np.squeeze(confidence)
            all_idx[i] = idx
    else:
        # We have chunked the orientations
        # Unpack the orientation information
        test_exp_maps, test_angles, test_rMat_ss, test_gvec_cs, test_rMat_c = orientation_data_to_test
        test_exp_maps = test_exp_maps[start:stop]
        test_angles = test_angles[start:stop]
        test_rMat_ss = test_rMat_ss[start:stop]
        test_gvec_cs = test_gvec_cs[start:stop]
        test_rMat_c = test_rMat_c[start:stop]
        # How many orientations?
        n_oris = np.shape(test_exp_maps)[0]
        # Initalize arrays
        all_exp_maps = np.zeros([n_coords, 3])
        all_confidence = np.zeros(n_coords)
        all_idx = np.zeros(n_coords, int)
        all_misorientation = np.zeros(n_coords)
        # Loop over the coordinates
        for i in np.arange(n_oris):
            orientation_data_to_test = [
                test_exp_maps[i], test_angles[i], test_rMat_ss[i], test_gvec_cs[i], test_rMat_c[i]]
            if n_coords == 1:
                exp_maps, confidence, dummy = _test_single_orientation_at_single_coordinate(
                    experiment, image_stack, coordinates_to_test, orientation_data_to_test)
            else:
                exp_maps, confidence = _test_single_orientation_at_many_coordinates(
                    experiment, image_stack, coordinates_to_test, orientation_data_to_test)
            # Replace any which are better than before
            to_replace = confidence > all_confidence
            all_exp_maps[to_replace, :] = exp_maps[to_replace]
            all_confidence[to_replace] = confidence[to_replace]
            all_idx[to_replace] = start + i

    # Refine if we need to, we have one orientation per coordinate point
    if refine_yes_no == 1:
        # Not sure if it is faster to pull the orientation info or compute it
        for i in np.arange(n_coords):
            coord_to_test = coordinates_to_test[i, :]
            single_orientation_data_to_test = _precompute_diffraction_data_of_single_orientation(
                experiment, all_exp_maps[i])
            exp_map, confidence, misorientation = _test_single_orientation_at_single_coordinate(
                experiment, image_stack, coord_to_test, single_orientation_data_to_test, refine_yes_no=1)
            # these come back as arrays with one element, which numpy will not put in a single element
            all_exp_maps[i] = np.squeeze(exp_map)
            all_confidence[i] = np.squeeze(confidence)
            all_misorientation[i] = np.squeeze(misorientation)

    return all_exp_maps, all_confidence, all_idx, all_misorientation, start, stop


def _test_orientation_bags_at_coordinates(experiment, image_stack, orientation_data, coordinates, bags, start=0, stop=0):
    """
        Goal:
            Test a different, small set of orientations ("bag") at each coordinate.
        Input:
            orientation_data: precomputed data (exp_maps, angles, rMat_ss, gvec_cs, rMat_c) of all candidate orientations
            coordinates: (n, 3) coordinates, only [start:stop] are tested
            bags: list of n arrays, bags[i] holds the indices into orientation_data to test at coordinate i
        Output:
            best_idx: index into orientation_data of the best orientation at each coordinate
            best_confidence: its confidence
            n_tests: number of orientation/coordinate tests performed
    """
    # Grab some experiment data
    tD = experiment.tVec_d  # Detector X,Y,Z translation (mm)
    rD = experiment.rMat_d  # Detector rotation matrix (rad)
    tS = experiment.tVec_s  # Sample X,Y,Z translation (mm)
    base = experiment.base
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals
    bsp = experiment.bsp
    ome_edges = experiment.ome_edges
    _, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c = orientation_data

    n_coords = stop - start
    best_idx = np.zeros(n_coords, int)
    best_confidence = np.zeros(n_coords)
    n_tests = 0
    for i in np.arange(n_coords):
        coord_to_test = coordinates[start + i]
        bag = bags[start + i]
        best = -1.0
        for ori in bag:
            det_xy = xfcapi.gvec_to_xy(
                all_gvec_cs[ori], rD, all_rMat_ss[ori], all_rMat_c[ori], tD, tS, coord_to_test)
            confidence = _quant_and_clip_confidence(det_xy, all_angles[ori][:, 2], image_stack,
                                                    base, inv_deltas, clip_vals, bsp, ome_edges)
            # Strictly greater, so a tie goes to the first orientation in the bag
            if confidence > best:
                best = confidence
                best_idx[i] = ori
        best_confidence[i] = best
        n_tests += len(bag)

    return best_idx, best_confidence, n_tests, start, stop


def _orientation_confidences(experiment, image_stack, coord, exp_maps):
    """Confidence at one coordinate of each orientation in exp_maps (n, 3).

    The diffraction data of every orientation is simulated here, so the
    orientations need not be precomputed.
    """
    exp_maps = np.atleast_2d(exp_maps)
    confidences = np.zeros(exp_maps.shape[0])
    for i, exp_map in enumerate(exp_maps):
        _, angles, rMat_ss, gvec_cs, rMat_c = \
            _precompute_diffraction_data_of_single_orientation(experiment, exp_map)
        det_xy = xfcapi.gvec_to_xy(
            gvec_cs, experiment.rMat_d, rMat_ss, np.squeeze(rMat_c),
            experiment.tVec_d, experiment.tVec_s, coord)
        confidences[i] = _quant_and_clip_confidence(
            det_xy, angles[:, 2], image_stack, experiment.base, experiment.inv_deltas,
            experiment.clip_vals, experiment.bsp, experiment.ome_edges)
    return confidences


def _grid_top_candidates(experiment, image_stack, grid_exp_maps, coords, n_keep, start=0, stop=0):
    """
        Goal:
            Score a chunk of an orientation grid at many coordinates and keep the best few per coordinate.
        Input:
            grid_exp_maps: (n, 3) exp-maps of the whole grid, only [start:stop] is used
            coords: (V, 3) coordinates to test
            n_keep: how many of the best grid orientations to keep per coordinate
        Output:
            top_idx: (V, k) indices into the whole grid
            top_confidence: (V, k) their confidences
            n_tests: number of orientation/coordinate tests performed
    """
    chunk = grid_exp_maps[start:stop]
    _, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c, _, _ = \
        _precompute_diffraction_data_of_many_orientations(experiment, chunk)
    n_oris = len(chunk)
    n_coords = len(coords)
    tD = experiment.tVec_d
    rD = experiment.rMat_d
    tS = experiment.tVec_s
    confidence = np.zeros((n_coords, n_oris))
    for i in np.arange(n_oris):
        for v in np.arange(n_coords):
            det_xy = xfcapi.gvec_to_xy(
                all_gvec_cs[i], rD, all_rMat_ss[i], all_rMat_c[i], tD, tS, coords[v])
            confidence[v, i] = _quant_and_clip_confidence(
                det_xy, all_angles[i][:, 2], image_stack, experiment.base,
                experiment.inv_deltas, experiment.clip_vals, experiment.bsp,
                experiment.ome_edges)
    k = min(n_keep, n_oris)
    top = np.argpartition(-confidence, k - 1, axis=1)[:, :k]
    return (start + top, np.take_along_axis(confidence, top, axis=1),
            n_oris*n_coords, start, stop)


def _refine_orientations_at_coordinates(experiment, image_stack, coordinates, start_exp_maps, radii_deg, start=0, stop=0):
    """
        Goal:
            Refine a starting orientation at each coordinate with the coarse-to-fine pattern search.
        Input:
            coordinates: (T, 3) coordinates, one per task
            start_exp_maps: (T, 3) orientation to start from at each coordinate
            radii_deg: search radii, see _pattern_search_refine
        Output:
            exp_maps: (stop - start, 3) refined orientations
            confidence: their confidences
            n_evals: number of orientations simulated and scored
    """
    n = stop - start
    exp_maps = np.zeros((n, 3))
    confidence = np.zeros(n)
    n_evals = 0
    for i in np.arange(n):
        coord = coordinates[start + i]
        counter = [0]

        def cost(trial_exp_maps):
            counter[0] += len(trial_exp_maps)
            return _orientation_confidences(experiment, image_stack, coord, trial_exp_maps)

        exp_maps[i], confidence[i] = _pattern_search_refine(
            start_exp_maps[start + i], cost, radii_deg)
        n_evals += counter[0]
    return exp_maps, confidence, n_evals, start, stop


def _precompute_diffraction_data_of_single_orientation(experiment, exp_map):
    """
        Goal:
            Read in one orientations and pre-compute all needed diffraction information on one CPU.
        Input:
            experiment: Packaged information holding experimental details
            controller: Packaged information holding multiprocessing details
            exp_maps_to_compute: exponential maps of each orientation to process
                Must of shape of (3) or (n_oris,3)
        Output:
            all_exp_maps: exponential maps of each orientation
                Will be of shape (n_oris,3)
            all_angles: list of eta, theta, omega data for each diffraction event of each orientation
                Will be of length = n_oris
            all_rMat_ss: list of sample rotation matrices for each diffraction event of each orientation
                Will be of length = n_oris
            all_gvec_cs: list of g-vectors for each diffraction event for each grain
                Will be of length = n_oris
            all_rMat_c: rotation matrices of each orientation
                Will be of shape (n_oris,3,3)
    """
    # Handle incoming size of exp_map
    exp_map = np.squeeze(exp_map)
    # Grab some experiment data
    # Packaged information about the material and HKLs
    plane_data = experiment.plane_data
    # Detector tilts, position, as well as stage position and chi [?,mm,mm,chi]
    detector_params = experiment.detector_params
    pixel_size = experiment.pixel_size  # Pixel size (mm)
    # Start and stop omega position of image stack (rad)
    ome_range = experiment.ome_range
    # Defined omega period for HEXRD to work in (rad)
    ome_period = experiment.ome_period
    # Pixels near the edge of the detector to avoid
    panel_dims_expanded = [(-10, -10), (10, 10)]
    # Assume grain is unstrained
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])

    # Transform exp_map to rotation matrix
    rMat_c = xfcapi.make_rmat_of_expmap(exp_map)
    # Define all parameters for the orientation (strain and orientation)
    gparams = np.hstack([exp_map, ref_gparams])
    # Simulate the the diffraction events
    sim_results = xrdutil.simulateGVecs(plane_data, detector_params, gparams, panel_dims=panel_dims_expanded,
                                        pixel_pitch=pixel_size, ome_range=ome_range, ome_period=ome_period,
                                        distortion=None)
    # Pull just the angles for each g-vector
    angles = sim_results[2]
    # Calculate the sample rotation matrix
    rMat_ss = xfcapi.make_sample_rmat(experiment.chi, angles[:, 2])
    # Convert the angles to g-vectors
    gvec_cs = xfcapi.angles_to_gvec(angles, chi=experiment.chi, rmat_c=rMat_c)
    # Handle arrays not being the correct size if we have only one orientation
    if len(np.shape(exp_map)) == 1:
        exp_map = np.expand_dims(exp_map, 0)
    if len(np.shape(rMat_c)) == 2:
        rMat_c = np.expand_dims(rMat_c, 0)
    # Return precomputed data
    return exp_map, angles, rMat_ss, gvec_cs, rMat_c


def _precompute_diffraction_data_of_many_orientations(experiment, exp_maps, start=0, stop=0):
    """
        Goal:
            Read in many orientations and pre-compute all needed diffraction information on one CPU.
        Input:
            experiment: Packaged information holding experimental details
            controller: Packaged information holding multiprocessing details
            exp_maps_to_compute: exponential maps of each orientation to process
                Must of shape of (n_oris,3)
        Output:
            all_exp_maps: exponential maps of each orientation
                Will be of shape (n_oris,3)
            all_angles: list of eta, theta, omega data for each diffraction event of each orientation
                Will be of length = n_oris
            all_rMat_ss: list of sample rotation matrices for each diffraction event of each orientation
                Will be of length = n_oris
            all_gvec_cs: list of g-vectors for each diffraction event for each grain
                Will be of length = n_oris
            all_rMat_c: rotation matrices of each orientation
                Will be of shape (n_oris,3,3)
    """

    # Grab some experiment data
    # Packaged information about the material and HKLs
    plane_data = experiment.plane_data
    # Detector tilts, position, as well as stage position and chi [?,mm,mm,chi]
    detector_params = experiment.detector_params
    pixel_size = experiment.pixel_size  # Pixel size (mm)
    # Start and stop omega position of image stack (rad)
    ome_range = experiment.ome_range
    # Defined omega period for HEXRD to work in (rad)
    ome_period = experiment.ome_period
    # Pixels near the edge of the detector to avoid
    panel_dims_expanded = [(-10, -10), (10, 10)]
    # Assume grain is unstrained
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])

    # How many orientations do we have?
    if start == 0 and stop == 0:
        # We are not in multiprocessing mode
        n_oris = np.shape(exp_maps)[0]
    else:
        # We are in multiprocessing mode and need to pull only some of the exp_maps
        exp_maps = exp_maps[start:stop, :]
        n_oris = np.shape(exp_maps)[0]

    # Initialize lists - these must be lists since some data is of a different size for each orientations
    all_exp_maps = [None] * n_oris
    all_angles = [None] * n_oris
    all_rMat_ss = [None] * n_oris
    all_gvec_cs = [None] * n_oris
    all_rMat_c = [None] * n_oris
    # Loop through the orientations and precompute information
    for i in np.arange(n_oris):
        # What orientation are we looking at?
        exp_map = exp_maps[i]
        # Transform exp_map to rotation matrix
        rMat_c = xfcapi.make_rmat_of_expmap(exp_map.T)
        # Define all parameters for the orientation (strain and orientation)
        gparams = np.hstack([exp_map, ref_gparams])
        # Simulate the the diffraction events
        sim_results = xrdutil.simulateGVecs(plane_data, detector_params, gparams, panel_dims=panel_dims_expanded,
                                            pixel_pitch=pixel_size, ome_range=ome_range, ome_period=ome_period,
                                            distortion=None)
        # Pull just the angles for each g-vector
        angles = sim_results[2]
        # Calculate the sample rotation matrix
        rMat_ss = xfcapi.make_sample_rmat(experiment.chi, angles[:, 2])
        # Convert the angles to g-vectors
        gvec_cs = xfcapi.angles_to_gvec(
            angles, chi=experiment.chi, rmat_c=rMat_c)
        # Drop data into arrays
        all_exp_maps[i] = exp_map
        all_angles[i] = angles
        all_rMat_ss[i] = rMat_ss
        all_gvec_cs[i] = gvec_cs
        all_rMat_c[i] = rMat_c

    # Return precomputed data
    return all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c, start, stop


def precompute_diffraction_data(experiment, controller, exp_maps_to_precompute):
    """
        Goal:
            Read in at least one orientation and pre-compute all needed diffraction information
                on as many CPUs as desired.
        Input:
            experiment: Packaged information holding experimental details
            controller: Packaged information holding multiprocessing details
            exp_maps_to_compute: exponential maps of each orientation to process
                Must of shape of either (3) or (n_oris,3)
        Output:
            all_exp_maps: exponential maps of each orientation
                Will be of shape (n_oris,3)
            all_angles: list of eta, theta, omega data for each diffraction event of each orientation
                Will be of length = n_oris
            all_rMat_ss: list of sample rotation matrices for each diffraction event of each orientation
                Will be of length = n_oris
            all_gvec_cs: list of g-vectors for each diffraction event for each grain
                Will be of length = n_oris
            all_rMat_c: rotation matrices of each orientation
                Will be of shape (n_oris,3,3)
    """
    # How many orientations?
    if len(np.shape(exp_maps_to_precompute)) == 1:
        exp_maps_to_precompute = np.expand_dims(exp_maps_to_precompute, 0)
    n_oris = np.shape(exp_maps_to_precompute)[0]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Are we dealing with one orientation or many?
    if n_oris == 1:
        # Single orientation, precompute the information
        # Tell the user what we are doing
        print('Precomputing diffraction data for 1 orientation on 1 CPU.')
        all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c = \
            _precompute_diffraction_data_of_single_orientation(
                experiment, exp_maps_to_precompute)
    else:
        # Many orientations, how many cores are we dealing with?
        if ncpus == 1:
            # Many orientations, single CPU
            # Tell the user what we are doing
            print(
                f'Precomputing diffraction data for {n_oris} orientations on 1 CPU.')
            all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c, start, stop = \
                _precompute_diffraction_data_of_many_orientations(
                    experiment, exp_maps_to_precompute)
        elif ncpus > 1:
            # Many orientations, many CPUs
            # Define the chunking
            chunks, starts, stops, chunk_size = make_chunks(
                n_oris, ncpus, controller.get_chunk_size())
            num_chunks = len(chunks)
            # Tell the user what we are doing
            print(
                f'Precomputing diffraction data for {n_oris} orientations on {ncpus} CPUs.')
            # Tell the user about the chunking
            print(
                f'There are {num_chunks} chunks with {chunk_size} orientations in each chunk.')
            # Initialize arrays to drop the precomputed data
            all_exp_maps = [None] * n_oris
            all_angles = [None] * n_oris
            all_rMat_ss = [None] * n_oris
            all_gvec_cs = [None] * n_oris
            all_rMat_c = [None] * n_oris
            # Package all inputs to the distributor function
            state = (starts, stops, experiment, exp_maps_to_precompute)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            with multiprocessing_pool(ncpus, state) as pool:
                for vals1, vals2, vals3, vals4, vals5, start, stop in pool.imap_unordered(_precompute_diffraction_data_distributor, chunks):
                    # Grab the data as each CPU drops it
                    all_exp_maps[start:stop] = vals1
                    all_angles[start:stop] = vals2
                    all_rMat_ss[start:stop] = vals3
                    all_gvec_cs[start:stop] = vals4
                    all_rMat_c[start:stop] = vals5
                    # Clean up
                    del vals1, vals2, vals3, vals4, vals5, start, stop
            # Final cleanup
            pool.close()
            pool.join()
        else:
            print('Number of CPUs must be 1 or greater.')

    # Handle arrays not being the correct size if we have only one orientation
    if len(np.shape(all_exp_maps)) == 1:
        all_exp_maps = np.expand_dims(all_exp_maps, 0)
    if len(np.shape(all_rMat_c)) == 2:
        all_rMat_c = np.expand_dims(all_rMat_c, 0)
    # Say we are done and return precomputed data
    print('Done precomputing orientation data.')
    return all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c


def test_orientations_at_coordinates(experiment, controller, image_stack, orientation_data_to_test, coordinates_to_test, refine_yes_no=0, return_misorientation=0):
    """
        Goal:

        Input:

        Output:

    """
    # How many orientations?
    n_oris = np.shape(orientation_data_to_test[0])[0]
    # How many coordinate points
    if len(np.shape(coordinates_to_test)) == 1:
        coordinates_to_test = np.expand_dims(coordinates_to_test, 0)
    n_coords = np.shape(coordinates_to_test)[0]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Start a timer
    t0 = timeit.default_timer()
    # Shoot a warning to the user if they are running with refinment
    if refine_yes_no == 1:
        print(
            f'Refining each orientation with a coarse-to-fine search over radii {REFINEMENT_RADII_DEG} degrees '
            '(27 trial orientations per pass, re-centered on the best while the confidence improves).')

    # What senario do we have?
    if ncpus == 1 or (n_oris == 1 and n_coords == 1):
        # Single processor variant
        # Tell the user what we are doing
        print(
            f'Testing {n_oris} orientations at {n_coords} coordinate points on 1 CPU.')
        # Treat entire coordinate array as a single chunk and run
        all_exp_maps, all_confidence, all_idx, all_misorientation, start, stop = _test_many_orientations_at_many_coordinates(
            experiment, image_stack, coordinates_to_test, orientation_data_to_test, refine_yes_no=refine_yes_no, start=0, stop=n_coords)
    else:
        if n_coords > n_oris:
            # Fastest to loop over orientations for single processor or chunk up coordinates for multiprocessor
            # Tell the user what we are doing
            print(
                f'Testing {n_oris} orientations at {n_coords} coordinate points on {ncpus} CPUs.')
            # Define the chunking
            chunks, starts, stops, chunk_size = make_chunks(
                n_coords, ncpus, controller.get_chunk_size())
            num_chunks = len(chunks)
            # Tell the user about the chunking
            print(f'The {ncpus} CPUs will tackle {num_chunks} chunks with {chunk_size} coordinate points to test against the {n_oris} orientations.')
            # Initialize arrays to drop the exp_map and confidence
            all_exp_maps = np.zeros([n_coords, 3])
            all_confidence = np.zeros(n_coords)
            all_idx = np.zeros(n_coords)
            all_misorientation = np.zeros(n_coords)
            # Unpack the orientation data
            # Package all inputs to the distributor function
            state = (starts, stops, experiment, image_stack,
                     coordinates_to_test, orientation_data_to_test, refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            with multiprocessing_pool(ncpus, state) as pool:
                for vals1, vals2, vals3, vals4, start, stop in pool.imap_unordered(_test_many_orientations_at_many_coordinates_distributor, chunks):
                    # Grab the data as each CPU drops it
                    all_exp_maps[start:stop] = vals1
                    all_confidence[start:stop] = vals2
                    all_idx[start:stop] = vals3
                    all_misorientation[start:stop] = vals4
                    # Clean up
                    del vals1, vals2, vals3, vals4, start, stop
            # Final cleanup
            pool.close()
            pool.join()
        else:
            # Fastest to loop over coordinates for single processor or chunk up orientations for multiprocessor
            # Tell the user what we are doing
            print(
                f'Testing {n_oris} orientations at {n_coords} coordinate points on {ncpus} CPUs.')
            # Define the chunking
            chunks, starts, stops, chunk_size = make_chunks(
                n_oris, ncpus, controller.get_chunk_size())
            num_chunks = len(chunks)
            # Tell the user about the chunking
            print(f'The {ncpus} CPUs will tackle {num_chunks} chunks with {chunk_size} orientations to test at the {n_coords} coordinate points.')
            # Initialize arrays to drop the exp_map and confidence
            all_exp_maps = np.zeros([n_coords, 3])
            all_confidence = np.zeros(n_coords)
            all_idx = np.zeros(n_coords)
            all_misorientation = np.zeros(n_coords)
            # Package all inputs to the distributor function
            state = (starts, stops, experiment, image_stack,
                     coordinates_to_test, orientation_data_to_test, refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            chunk_num = 0
            with multiprocessing_pool(ncpus, state) as pool:
                for vals1, vals2, vals3, vals4, start, stop in pool.imap_unordered(_test_many_orientations_at_many_coordinates_distributor, chunks):
                    # Grab the data as each CPU drops it
                    # Replace any which are better than before
                    to_replace = vals2 > all_confidence
                    all_exp_maps[to_replace, :] = vals1[to_replace]
                    all_confidence[to_replace] = vals2[to_replace]
                    all_idx[to_replace] = vals3[to_replace]
                    all_misorientation[to_replace] = vals4[to_replace]
                    # Clean up
                    chunk_num = chunk_num + 1
                    del vals1, vals2, vals3, vals4, start, stop
            # Final cleanup
            pool.close()
            pool.join()

    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(f'Completed {n_oris*n_coords} orientation/coordinate tests in {np.round(elapsed, 1)} seconds ({elapsed/n_oris/n_coords} seconds per test).')
    else:
        print(f'Completed {n_oris*n_coords} orientation/coordinate tests in {np.round(elapsed/60, 1)} minutes ({elapsed/n_oris/n_coords} seconds per test).')

    if return_misorientation == 1:
        # Spit out the misorientation as well
        return all_exp_maps, all_confidence, all_idx.astype(int), all_misorientation
    else:
        return all_exp_maps, all_confidence, all_idx.astype(int)


def load_all_images(configuration, filenames, controller):
    """
        Goal:

        Input:

        Output:

    """
    if configuration.images.loading.load_style == 'h5':
        # Large h5 file read
        num_h5s = len(filenames)
        nframes = configuration.images.loading.nframes
        num_imgs = nframes*num_h5s
        # Load up a single file to get the shape and data type
        with h5py.File(filenames[0], 'r') as data:
            mini_image_stack = data['entry/data/data/'][:]
            image_shape = np.shape(mini_image_stack)[
                1:3]  # Don't need the first
            image_dtype = mini_image_stack.dtype
        raw_image_stack = np.zeros(
            [num_imgs, image_shape[0], image_shape[1]], image_dtype)
        start = 0
        for scan in np.arange(num_h5s):
            # Load up the data
            with h5py.File(filenames[scan], 'r') as data:
                raw_image_stack[start:start +
                                nframes] = data['entry/data/data/'][:]
            start = start + nframes

    else:
        # Singe image read with multiprocessing
        # Start a timer
        t0 = timeit.default_timer()
        # How many images to load
        n_imgs = len(filenames)
        # How many CPUs?
        ncpus = controller.get_process_count()
        # Grab in formation about the files
        quick_image = skimage.io.imread(filenames[0])
        image_shape = np.shape(quick_image)
        image_dtype = quick_image.dtype
        # Single process or multi-thread?
        if ncpus == 1:
            # Just go ahead and load the images
            print(f'Loading {n_imgs} images with a single CPU.')
            raw_image_stack, start, stop = _load_images(
                filenames, image_shape, image_dtype, 0, n_imgs)
        else:
            # Generate the blank image stack
            raw_image_stack = np.zeros(
                [n_imgs, image_shape[0], image_shape[1]], image_dtype)
            # Define the chunking
            chunks, starts, stops, chunk_size = make_chunks(
                n_imgs, ncpus, controller.get_chunk_size())
            num_chunks = len(chunks)
            print(
                f'Loading {n_imgs} images with {ncpus} CPUs and {num_chunks} chunks of size {chunk_size}.')
            # Package all inputs to the distributor function
            state = (starts, stops, filenames, image_shape, image_dtype)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            with multiprocessing_pool(ncpus, state) as pool:
                for vals1, start, stop in pool.imap_unordered(_load_images_distributor, chunks):
                    # Grab the data as each CPU drops it
                    raw_image_stack[start:stop, :, :] = vals1
                    # Clean up
                    del vals1, start, stop

        # How long did it take?
        t1 = timeit.default_timer()
        elapsed = t1-t0
        if elapsed < 60.0:
            print(
                f'Loaded {n_imgs} images in {np.round(elapsed, 1)} seconds ({elapsed/n_imgs} seconds per image).')
        else:
            print(
                f'Loaded {n_imgs} images in {np.round(elapsed/60, 1)} minutes ({elapsed/n_imgs} seconds per image).')

    return raw_image_stack


def remove_median_darkfields(raw_image_stack, controller, configuration):
    """
        Goal:

        Input:

        Output:

    """
    # Start a timer
    t0 = timeit.default_timer()
    # How many slices are we dealing with
    n_slices = np.shape(raw_image_stack)[2]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Pull configuration data
    median_size_through_omega = configuration.images.processing.omega_kernel_size
    global_threshold = configuration.images.processing.threshold
    # Single process or multi-thread?
    if ncpus == 1:
        # Just go ahead and load the images
        print(
            f'Subtracting dynamic darkfield from {n_slices} slices with a single CPU.')
        cleaned_image_stack, start, stop = _remove_dynamic_median(
            raw_image_stack, median_size_through_omega, 0, n_slices)
    else:
        # Generate the blank image stack
        cleaned_image_stack = np.zeros(
            np.shape(raw_image_stack), raw_image_stack.dtype)
        # Define the chunking
        chunks, starts, stops, chunk_size = make_chunks(
            n_slices, ncpus, controller.get_chunk_size())
        num_chunks = len(chunks)
        print(
            f'Subtracting dynamic darkfield from {n_slices} slices with {ncpus} CPUs and {num_chunks} chunks of size {chunk_size}.')
        # Package all inputs to the distributor function
        state = (starts, stops, raw_image_stack, median_size_through_omega)
        # print(multiprocessing_pool)
        # Start the multiprocessing loop
        try:
            set_multiprocessing_method(controller.multiprocessing_start_method)
        except:
            print('multiprocessing method already set')

        with multiprocessing_pool(ncpus, state) as pool:
            for vals1, start, stop in pool.imap_unordered(_remove_median_darkfield_distributor, chunks):
                # Grab the data as each CPU drops it
                cleaned_image_stack[:, :, start:stop] = vals1
                # Clean up
                del vals1, start, stop

    # Remove the global threshold
    if global_threshold > 0:
        print('Dynamic darkfield generated and subtracted.')
        print(f'Subtracting global threshold of {global_threshold}.')
        mask = cleaned_image_stack <= global_threshold
        cleaned_image_stack[mask] = 0
        cleaned_image_stack[~mask] = cleaned_image_stack[~mask] - \
            global_threshold

    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(
            f'Subtracted dynamic darkfield and global threshold from {n_slices} slices in {np.round(elapsed, 1)} seconds ({elapsed/n_slices} seconds per slice).')
    else:
        print(
            f'Subtracted dynamic darkfield and global threshold from {n_slices} slices in {np.round(elapsed/60, 1)} minutes ({elapsed/n_slices} seconds per slice).')

    return cleaned_image_stack


def filter_and_binarize_images(cleaned_image_stack, controller, filter_parameters):
    """
        Goal:

        Input:

        Output:

    """
    # Start a timer
    t0 = timeit.default_timer()
    # How many slices are we dealing with
    n_images = np.shape(cleaned_image_stack)[0]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Create some text
    cleanup_text = ['Gaussian Cleanup',
                    'Errosion/Dilation Cleanup', 'Non-Local Means Cleanup']
    small_objects_text = [
        'no Filtering of Small Objects', 'Filtering of Small Objects']
    # Filter Parameters information
    # filter_parameters[0] - if 1, remove small objects, if 0 do nothing
    # filter_parameters[1] - what size of small objects to remove
    # filter_parameters[2] - which cleanup to use
    # filter_parameters[3:] - cleanup parameters
    # Single process or multi-thread?
    if ncpus == 1:
        # Just go ahead and load the images
        print(
            f'Cleaning {n_images} images with {cleanup_text[filter_parameters[2]]} and {small_objects_text[filter_parameters[0]]} on a single CPU.')
        binarized_image_stack, start, stop = _filter_and_binarize_image(
            cleaned_image_stack, filter_parameters, 0, n_images)
    else:
        # Generate the blank image stack
        binarized_image_stack = np.zeros(np.shape(cleaned_image_stack), bool)
        # Define the chunking
        chunks, starts, stops, chunk_size = make_chunks(
            n_images, ncpus, controller.get_chunk_size())
        num_chunks = len(chunks)
        print(f'Cleaning {n_images} images with {cleanup_text[filter_parameters[2]]} and {small_objects_text[filter_parameters[0]]}\n\
              with {ncpus} CPUs and {num_chunks} chunks of size {chunk_size}')
        # Package all inputs to the distributor function
        state = (starts, stops, cleaned_image_stack, filter_parameters)

        # Start the multiprocessing loop
        set_multiprocessing_method(controller.multiprocessing_start_method)
        with multiprocessing_pool(ncpus, state) as pool:
            for vals1, start, stop in pool.imap_unordered(_filter_and_binarize_images_distributor, chunks):
                # Grab the data as each CPU drops it
                binarized_image_stack[start:stop, :, :] = vals1
                # Clean up
                del vals1, start, stop

    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(
            f'Filtered {n_images} images in {np.round(elapsed, 1)} seconds ({elapsed/n_images} seconds per image).')
    else:
        print(
            f'Filtered {n_images} images in {np.round(elapsed/60, 1)} minutes ({elapsed/n_images} seconds per image).')

    return binarized_image_stack


def _run_chunk(func, chunk):
    # _mp_state = (starts, stops, *args for func), set by multiprocessing_pool
    starts = _mp_state[0]
    stops = _mp_state[1]
    return func(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])


def _precompute_diffraction_data_distributor(chunk):
    return _run_chunk(_precompute_diffraction_data_of_many_orientations, chunk)


def _test_many_orientations_at_many_coordinates_distributor(chunk):
    return _run_chunk(_test_many_orientations_at_many_coordinates, chunk)


def _load_images_distributor(chunk):
    return _run_chunk(_load_images, chunk)


def _test_orientation_bags_at_coordinates_distributor(chunk):
    return _run_chunk(_test_orientation_bags_at_coordinates, chunk)


def _grid_top_candidates_distributor(chunk):
    return _run_chunk(_grid_top_candidates, chunk)


def _refine_orientations_distributor(chunk):
    return _run_chunk(_refine_orientations_at_coordinates, chunk)


def _remove_median_darkfield_distributor(chunk):
    return _run_chunk(_remove_dynamic_median, chunk)


def _filter_and_binarize_images_distributor(chunk):
    return _run_chunk(_filter_and_binarize_image, chunk)


def _remove_dynamic_median(raw_image_stack, median_size_through_omega=25, start=0, stop=0):
    # How many slices?
    n_slices = stop - start
    # Create a new variable with just want we need so we are not pinning raw_image_stack so much
    cleaned_slices = np.copy(raw_image_stack[:, :, start:stop])
    for slice in np.arange(n_slices):
        # Grab a slice at a specific through the raw image stack (does not really matter which axis)
        raw_slice = cleaned_slices[:, :, slice]
        # Calculate a moving median along omega at each
        raw_slice_dark = scipy.ndimage.median_filter(
            raw_slice, size=[median_size_through_omega, 1])
        # Update the slice and handle negatives
        new_slice = raw_slice - raw_slice_dark
        new_slice[raw_slice_dark >= raw_slice] = 0
        # Place in slices
        cleaned_slices[:, :, slice] = new_slice
    # Return the things
    return cleaned_slices, start, stop


def _filter_and_binarize_image(cleaned_image_stack, filter_parameters, start, stop):
    # Grab a chunk of the image stack
    working_image_stack = np.copy(cleaned_image_stack[start:stop, :, :])
    # Create a binarized image stack
    binarized_image_stack = np.zeros(np.shape(working_image_stack), bool)
    # How many images?
    n_images = np.shape(working_image_stack)[0]
    # Filter Parameters information
    # filter_parameters[0] - if 1, remove small objects, if 0 do nothing
    # filter_parameters[1] - what size of small objects to remove
    # filter_parameters[2] - which cleanup to use
    # filter_parameters[3:] - cleanup parameters
    # What filter are we using?
    which_filter = filter_parameters[2]
    if which_filter == 0:
        # Gaussian cleanup
        # Grab parameters
        [sigma, threshold] = filter_parameters[3:]
        for i in np.arange(n_images):
            # Grab the image
            img = working_image_stack[i, :, :]
            # Filter
            img = skimage.filters.gaussian(
                img, sigma=sigma, preserve_range=True)
            # Threshold and put into binary stack
            binarized_image_stack[i, :, :] = img > threshold
    elif which_filter == 1:
        # Errosion/dilation cleanup
        # Grab parameters
        [errosions, dilations, threshold] = filter_parameters[3:]
        for i in np.arange(n_images):
            # Grab the image
            img = working_image_stack[i, :, :]
            # Binarize
            img_binary = img > threshold
            # Errode then dilate
            img_binary = scipy.ndimage.binary_erosion(
                img_binary, iterations=errosions)
            img_binary = scipy.ndimage.binary_dilation(
                img_binary, iterations=dilations)
            # Toss into binary stack
            binarized_image_stack[i, :, :] = img_binary
    elif which_filter == 2:
        # Non-local means cleanup
        # Grab parameters
        [patch_size, patch_distance, threshold] = filter_parameters[3:]
        for i in np.arange(n_images):
            # Grab the image
            img = working_image_stack[i, :, :]
            # Estimage the per-slice sigma
            s_est = skimage.restoration.estimate_sigma(img)
            # Run non-local_means
            img = skimage.restoration.denoise_nl_means(
                img, sigma=s_est, h=0.8 * s_est, patch_size=patch_size, patch_distance=patch_distance, preserve_range=True)
            # Binarize and throw into new stack
            binarized_image_stack[i, :, :] = img > threshold
    elif which_filter == 3:
        # Grab parameters
        [sigma, median_filter_size] = filter_parameters[3:]
        for i in np.arange(n_images):
            img = working_image_stack[i, :, :]
            g = gaussian_filter(img, sigma=sigma)
            med = median_filter(g, size=median_filter_size)

            binarized_image_stack[i, :, :] = med > 0  # cleaned
    # Are we removing small features?
    remove_small_features = filter_parameters[0]
    if remove_small_features == 1:
        for i in np.arange(n_images):
            binarized_image_stack[i, :, :] = skimage.morphology.remove_small_objects(
                binarized_image_stack[i, :, :], filter_parameters[1], connectivity=1)

    # Return the things
    return binarized_image_stack, start, stop


def dilate_image_stack(binarized_image_stack, dilate_omega):
    if dilate_omega > 0:
        # Start a timer
        t0 = timeit.default_timer()
        # Tell the user
        print('Dilating image stack.')
        dilated_image_stack = scipy.ndimage.binary_dilation(
            binarized_image_stack, iterations=dilate_omega)
        # How long did it take?
        t1 = timeit.default_timer()
        elapsed = t1-t0
        if elapsed < 60.0:
            print(f'Dilated image stack in {np.round(elapsed, 1)} seconds.')
        else:
            print(f'Dilated image stack in {np.round(elapsed/60, 1)} minutes.')

        # Return the thing
        return dilated_image_stack
    else:
        print('No dilation asked for, returning binarized image stack.')
        return binarized_image_stack
