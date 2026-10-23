"""Readers and writers for NF data, metadata and image stacks.

contributing authors: dcp5303, ken38, seg246, Austin Gerlt, Simon Mason, lim37
"""

import glob
import h5py
import json
import numpy as np
import os
import pandas as pd
import re
import skimage

from .graphics import generate_ori_map

def write_to_h5(file_dir, file_name, data_array, data_name):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

    file_string = os.path.join(file_dir, file_name) + '.h5'
    hf = h5py.File(file_string, 'a')
    hf.require_dataset(data_name, np.shape(data_array), dtype=data_array.dtype)
    data = hf[data_name]
    data[...] = data_array
    hf.close()


def xmdf_writer(file_dir, file_name):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

    hf = h5py.File(os.path.join(file_dir, file_name)+'.h5', 'r')
    k = list(hf.keys())
    totalsets = len(k)
    # What are the datatypes
    datatypes = np.empty([totalsets], dtype=object)
    databytes = np.zeros([totalsets], dtype=np.int8)
    # note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))
    dims = np.ones([totalsets, 4], dtype=int)
    for i in np.arange(totalsets):
        datatypes[i] = hf[k[i]].dtype
        databytes[i] = hf[k[i]].dtype.itemsize
        s = np.shape(hf[k[i]])
        if len(s) > 4:
            print(
                'An array has greater than 4 dimensions - this writer cannot handle that')
            hf.close()
        dims[i, 0:len(s)] = s

    hf.close()

    filename = os.path.join(file_dir, file_name) + '.xdmf'
    f = open(filename, 'w')

    # Header for xml file
    f.write('<?xml version="1.0"?>\n')
    f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"[]>\n')
    f.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
    f.write(' <Domain>\n')
    f.write('\n')
    f.write('  <Grid Name="Cell Data" GridType="Uniform">\n')
    f.write('    <Topology TopologyType="3DCoRectMesh" Dimensions="' +
            str(dims[0, 0]+1) + ' ' + str(dims[0, 1]+1) + ' ' + str(dims[0, 2]+1) + '"></Topology>\n')
    f.write('    <Geometry Type="ORIGIN_DXDYDZ">\n')
    f.write('      <!-- Origin -->\n')
    f.write('      <DataItem Format="XML" Dimensions="3">0 0 0</DataItem>\n')
    f.write('      <!-- DxDyDz (Spacing/Resolution)-->\n')
    f.write('      <DataItem Format="XML" Dimensions="3">1 1 1</DataItem>\n')
    f.write('    </Geometry>\n')
    f.write('\n')

    for i in np.arange(totalsets):
        if dims[i, 3] == 1:
            f.write('      <Attribute Name="' +
                    k[i] + '" AttributeType="Scalar" Center="Cell">\n')
            f.write('      <DataItem Format="HDF" Dimensions="' + str(dims[i, 0]) + ' ' + str(dims[i, 1]) + ' ' + str(
                dims[i, 2]) + ' ' + str(dims[i, 3]) + '" NumberType="' + str(datatypes[i]) + '" Precision="' + str(databytes[i]) + '" >\n')
            f.write('       ' + file_name + '.h5:/' + k[i] + '\n')
            f.write('      </DataItem>\n')
            f.write('       </Attribute>\n')
            f.write('\n')
        elif dims[i, 3] == 3:
            f.write('      <Attribute Name="' +
                    k[i] + '" AttributeType="Vector" Center="Cell">\n')
            f.write('      <DataItem Format="HDF" Dimensions="' + str(dims[i, 0]) + ' ' + str(dims[i, 1]) + ' ' + str(
                dims[i, 2]) + ' ' + str(dims[i, 3]) + '" NumberType="' + str(datatypes[i]) + '" Precision="' + str(databytes[i]) + '" >\n')
            f.write('       ' + file_name + '.h5:/' + k[i] + '\n')
            f.write('      </DataItem>\n')
            f.write('       </Attribute>\n')
            f.write('\n')
        else:
            print('Wrong array size of ' +
                  str(dims[i, :]) + ' in array ' + str(i))

    # End the xmf file
    f.write('  </Grid>\n')
    f.write(' </Domain>\n')
    f.write('</Xdmf>\n')

    f.close()


def save_nf_data(save_dir, save_stem, grain_map, confidence_map, Xs, Ys, Zs, ori_list, tomo_mask=None, id_remap=None, save_type=['npz']):

    # H5 functionality added by SEG 8/3/2023

    print('Saving grain map data...')
    if id_remap is not None:
        if save_type[0] == 'hdf5':
            print('Writing HDF5 data...')
            save_string = os.path.join(save_dir, save_stem) + '_grain_map_data.h5'
            hf = h5py.File(save_string, 'w')
            hf.create_dataset('grain_map', data=grain_map)
            hf.create_dataset('confidence', data=confidence_map)
            hf.create_dataset('Xs', data=Xs)
            hf.create_dataset('Ys', data=Ys)
            hf.create_dataset('Zs', data=Zs)
            hf.create_dataset('id_remap', data=id_remap)
            if tomo_mask is not None:
                hf.create_dataset('tomo_mask', data=tomo_mask)
            hf.close()

        elif save_type[0] == 'npz':
            print('Writing NPZ data...')
            if tomo_mask is not None:
                np.savez(os.path.join(save_dir, save_stem) + '_grain_map_data.npz', grain_map=grain_map, confidence_map=confidence_map,
                         Xs=Xs, Ys=Ys, Zs=Zs, ori_list=ori_list, id_remap=id_remap, tomo_mask=tomo_mask)
            else:
                np.savez(os.path.join(save_dir, save_stem) + '_grain_map_data.npz', grain_map=grain_map,
                         confidence_map=confidence_map, Xs=Xs, Ys=Ys, Zs=Zs, ori_list=ori_list, id_remap=id_remap)

    else:
        if save_type[0] == 'hdf5':
            print('Writing HDF5 data...')
            save_string = os.path.join(save_dir, save_stem) + '_grain_map_data.h5'
            hf = h5py.File(save_string, 'w')
            hf.create_dataset('grain_map', data=grain_map)
            hf.create_dataset('confidence', data=confidence_map)
            hf.create_dataset('Xs', data=Xs)
            hf.create_dataset('Ys', data=Ys)
            hf.create_dataset('Zs', data=Zs)
            if tomo_mask is not None:
                hf.create_dataset('tomo_mask', data=tomo_mask)
            hf.close()

        elif save_type[0] == 'npz':
            print('Writing NPZ data...')
            if tomo_mask is not None:
                np.savez(os.path.join(save_dir, save_stem) + '_grain_map_data.npz', grain_map=grain_map,
                         confidence_map=confidence_map, Xs=Xs, Ys=Ys, Zs=Zs, ori_list=ori_list, tomo_mask=tomo_mask)
            else:
                np.savez(os.path.join(save_dir, save_stem) + '_grain_map_data.npz', grain_map=grain_map,
                         confidence_map=confidence_map, Xs=Xs, Ys=Ys, Zs=Zs, ori_list=ori_list)


def save_nf_data_for_paraview(file_dir, file_stem, grain_map, confidence_map, Xs, Ys, Zs, ori_list, mat, tomo_mask=None, id_remap=None, diffraction_volume_number=None, misorientation_map=None):

    print('Writing HDF5 data...')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(confidence_map, [1, 0, 2]), [2, 1, 0]), 'confidence')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(grain_map, [1, 0, 2]), [2, 1, 0]), 'grain_map')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(Xs, [1, 0, 2]), [2, 1, 0]), 'Xs')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(Ys, [1, 0, 2]), [2, 1, 0]), 'Ys')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(Zs, [1, 0, 2]), [2, 1, 0]), 'Zs')
    # Check for tomo mask
    if tomo_mask is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(tomo_mask, [1, 0, 2]), [2, 1, 0]), 'tomo_mask')
    # Check for diffraction volume numbers
    if diffraction_volume_number is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data', np.transpose(np.transpose(
            diffraction_volume_number, [1, 0, 2]), [2, 1, 0]), 'diffraction_volume_number')
    # Write the misorientaiton if it exits
    if misorientation_map is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data', np.transpose(
            np.transpose(misorientation_map, [1, 0, 2]), [2, 1, 0]), 'misorientation_map')
    # Create IPF colors
    # From unitcel the color is in hsl format
    rgb_image = generate_ori_map(grain_map, ori_list, mat, id_remap)
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(rgb_image, [1, 0, 2, 3]), [2, 1, 0, 3]), 'IPF_010')
    print('Writing XDMF...')
    xmdf_writer(file_dir, file_stem + '_grain_map_data')
    print('All done writing.')


def generate_filepaths_and_omegas(configuration, downselection_number=None):
    if configuration.images.loading.load_style == 'par':
        # Comb the nf folder for metadata files (.json and .par) and compile them
        all_meta = skim_metadata(configuration)

        # Find the folders associated with this z_height
        meta = all_meta[np.round(all_meta[configuration.images.loading.vertical_motor_name], 5)
                        == configuration.images.loading.target_vertical_position]

        # Grab the array of per-frame omega values and file locations
        filenames, num_imgs = skim_image_locations(
            meta, configuration.images.loading.sample_raw_data_folder)

        # Generate the omega edges from the .par file information
        _, omega_edges_deg = generate_omega_edges(meta, num_imgs)

    elif configuration.images.loading.load_style == 'direct':
        # Load configuation manually
        filenames, num_imgs = generate_image_locations(configuration)
        omega_edges_deg = np.linspace(
            configuration.experiment.omega_start, configuration.experiment.omega_stop, num_imgs+1)

    elif configuration.images.loading.load_style == 'h5':
        # Pull data from config
        raw_data_path = configuration.images.loading.sample_raw_data_folder
        folders = configuration.images.loading.data_folders
        omega_starts = configuration.images.loading.omega_starts
        omega_stops = configuration.images.loading.omega_stops
        nframes = configuration.images.loading.nframes
        nscans = len(folders)

        filenames = []
        start = 0
        for folder in np.arange(nscans):
            filenames.append(glob.glob(raw_data_path + os.sep +
                             str(folders[folder]) + "/nf/*.h5")[0])

        if omega_starts is not None:
            # Splice together the omegas - TODO: Make the final image in each stack not too large...
            start = 0
            omega_edges_deg = np.zeros(nframes*nscans+1)
            for folder in np.arange(nscans):
                omega_edges_deg[start:start+nframes+1] = np.linspace(
                    omega_starts[folder], omega_stops[folder], nframes+1)
                start = start+nframes
                # This will overwrite the end of the previous set of scans to lengthen the final image of each chunk - see TODO above
        else:
            # Default to original
            num_imgs = nframes*nscans
            omega_edges_deg = np.linspace(
                configuration.experiment.omega_start, configuration.experiment.omega_stop, num_imgs+1)

    if downselection_number is not None:
        filenames = filenames[0:downselection_number]
        omega_edges_deg = omega_edges_deg[0:downselection_number+1]

    return filenames, omega_edges_deg


def load_images_from_npz(experiment):
    analysis_name = experiment.config.analysis_name
    output_directory = experiment.config.output_directory
    filename = output_directory + os.sep + \
        analysis_name + '_packaged_images_and_omegas.npz'
    data = np.load(filename, mmap_mode="r")

    image_stack = data['image_stack']
    print(f'Images loaded from: {filename}')
    nframes = np.shape(image_stack)[0]

    # Update the experiment
    experiment.nframes = nframes

    return experiment, image_stack


def load_omegas_from_npz_and_correct(experiment, omega_shift_deg=None, rotation_range=None):
    analysis_name = experiment.config.analysis_name
    output_directory = experiment.config.output_directory
    filename = output_directory + os.sep + \
        analysis_name + '_packaged_images_and_omegas.npz'
    data = np.load(filename, mmap_mode="r")

    omega_edges_deg = data['omega_edges_deg']
    print(f'Omegas loaded from: {filename}')
    if rotation_range is not None:
        omega_edges_deg = omega_edges_deg*(rotation_range/360.)

    # Do we have a shift in omega
    if omega_shift_deg is not None:
        # Shift the omegas
        omega_edges_deg = omega_edges_deg + omega_shift_deg
        print(f'Omegas corrected by: {omega_shift_deg} degrees.')
    print(
        f'Rotation range: {omega_edges_deg[0]}-{omega_edges_deg[-1]} degrees.')

    # Define omega edges in radians
    ome_edges = omega_edges_deg*np.pi/180

    # Define variables in degrees
    # Omega range is the experimental span of omega space
    ome_range_deg = [
        (omega_edges_deg[0], omega_edges_deg[experiment.nframes])]  # Degrees
    # Omega period is the range in which your omega space lies (often 0 to 360 or -180 to 180)
    ome_period_deg = (ome_range_deg[0][0], ome_range_deg[0][0]+360.)  # Degrees
    # Define variables in radians
    ome_period = (ome_period_deg[0]*np.pi/180., ome_period_deg[1]*np.pi/180.)
    ome_range = [(ome_range_deg[0][0]*np.pi/180.,
                  ome_range_deg[0][1]*np.pi/180.)]
    x_col_edges = experiment.x_col_edges
    y_row_edges = experiment.y_row_edges
    base = np.array([x_col_edges[0],
                     y_row_edges[0],
                     ome_edges[0]])
    deltas = np.array([x_col_edges[1] - x_col_edges[0],
                       y_row_edges[1] - y_row_edges[0],
                       ome_edges[1] - ome_edges[0]])
    inv_deltas = 1.0/deltas

    # Update the experiment
    experiment.ome_range = ome_range
    experiment.ome_period = ome_period
    experiment.ome_edges = ome_edges
    experiment.base = base
    experiment.inv_deltas = inv_deltas
    experiment.omega_correction = omega_shift_deg

    return experiment


def skim_metadata(configuration):
    """
    skims all the .josn and .par files in a folder, and returns a concacted
    pandas DataFrame object with duplicates removed. If Dataframe=False, will
    return the same thing but as a dictionary of dictionaries.
    NOTE: uses Pandas Dataframes because some data is int, some float, some
    string. Pandas auto-parses dtypes per-column, and also has
    dictionary-like indexing.
    """
    # Pull folder name from the configuation
    nf_raw_folder = configuration.images.loading.sample_raw_data_folder
    json_and_par_starter = configuration.images.loading.json_and_par_starter
    raw_folder = nf_raw_folder + os.sep + json_and_par_starter

    # Grab all the nf json files, assert they both exist and have par pairs
    f_jsons = glob.glob(raw_folder + "*json")
    assert len(f_jsons) > 0, "No .jsons found in {}".format(raw_folder)
    f_par = [x[:-4] + "par" for x in f_jsons]
    assert np.all([os.path.isfile(x) for x in f_par]), "missing .par files"

    # Read in headers from jsons
    headers = [json.load(open(j, "r")).values() for j in f_jsons]

    # Read in headers from each json and data from each par as Dataframes
    df_list = [
        pd.read_csv(p, names=h, sep=r'\s+', comment="#")
        for h, p in zip(headers, f_par)
    ]

    # Concactionate into a single dataframe and delete duplicate columns
    meta_df_dups = pd.concat(df_list, axis=1)
    meta_df = meta_df_dups.loc[:, ~meta_df_dups.columns.duplicated()].copy()

    return meta_df


def skim_image_locations(meta_df, raw_folder):
    """ takes in a dataframe generated from the metadata using "skim_metadata"
    plus the near_field folder locations, and returns a list of image locations
    """
    scan = meta_df['SCAN_N'].to_numpy()
    first = meta_df['goodstart'].to_numpy()
    num_frames_anticipated = meta_df['nframes_real'].to_numpy()
    num_imgs_per_scan = np.zeros(len(scan), dtype=int)

    files = []
    flag = 0
    for i in range(len(scan)):
        nf_folder = raw_folder + os.sep + str(scan[i]) + "/nf/"
        all_files = glob.glob(nf_folder + "*.tif") + \
            glob.glob(nf_folder + "*.tiff")
        all_names = [x.split(os.sep)[-1] for x in all_files]
        all_ids_list = [int(re.findall("([0-9]+)", x)[0]) for x in all_names]
        all_ids = np.array(all_ids_list)
        good_ids = (all_ids >= first[i]) * (all_ids <=
                                            first[i]+num_frames_anticipated[i]-1)
        files.append([x for x, y in sorted(zip(all_files, good_ids)) if y])
        if sum(good_ids) != num_frames_anticipated[i]:
            flag = 1
            print('There are ' + str(sum(good_ids)) + ' images in scan ' + str(scan[i]) +
                  ' when ' + str(num_frames_anticipated[i]) + ' were expected')
        num_imgs_per_scan[i] = sum(good_ids)

    if flag == 1:
        print("HEY, LISTEN!  There was an unexpected number of images within at least one scan folder.\n\
        This code will proceed with the shortened number of images and will assume that\n\
        the first image is still the 'goodstart' as defined in the par file.")

    # flatten the list of lists
    files = [item for sub in files for item in sub]
    # sanity check
    s_id = np.array(
        [int(re.findall("([0-9]+)", x.split(os.sep)[-3])[0]) for x in files]
    )
    f_id = np.array(
        [int(re.findall("([0-9]+)", x.split(os.sep)[-1])[0]) for x in files]
    )
    assert np.all(s_id[1:] - s_id[:-1] >= 0), "folders are out of order"
    assert np.all(f_id[1:] - f_id[:-1] >= 0), "files are out of order"

    return files, num_imgs_per_scan


def generate_image_locations(configuration):
    # Pull data from config
    num_images_per_folder = configuration.images.loading.nframes
    img_stem = configuration.images.loading.stem
    num_digits = configuration.images.loading.num_digits
    raw_data_path = configuration.images.loading.sample_raw_data_folder
    folders = configuration.images.loading.data_folders
    goodstarts = configuration.images.loading.goodstarts

    num_folders = len(folders)
    filenames = [None]*(num_images_per_folder*num_folders)
    img = 0
    for folder in np.arange(num_folders):
        nf_folder = f'{raw_data_path}/{folders[folder]}/nf'
        # Files may end in either .tif or .tiff; check the first image
        first_img = f'{nf_folder}/{img_stem}{str(goodstarts[folder]).zfill(num_digits)}'
        ext = next((e for e in ('.tif', '.tiff')
                    if os.path.exists(first_img + e)), None)
        if ext is None:
            raise FileNotFoundError(
                f'Could not find {first_img} with either .tif or .tiff')
        for folder_img in np.arange(num_images_per_folder):
            filenames[img] = f'{nf_folder}/{img_stem}{str(goodstarts[folder]+folder_img).zfill(num_digits)}{ext}'
            img = img + 1

    return filenames, num_images_per_folder*num_folders


def generate_omega_edges(meta_df, num_imgs_per_scan):
    """ takes in a dataframe generated from the metadata using "skim_metadata",
    and returns a numpy array of the omega data (ie, what frames represent
    which omega angle in the results)"""
    start = meta_df["ome_start_real"].to_numpy(
    )  # ome_start_real is the starting omega position of the first good frame's omega slew
    # ome_end_read is the final omega position of the last good frame's omega slew
    stop = meta_df["ome_end_real"].to_numpy()
    steps = num_imgs_per_scan
    num_frames_anticipated = meta_df['nframes_real'].to_numpy()
    # This *should* not be needed if meta data was written correctly
    if np.sum(num_frames_anticipated-steps) != 0:
        print('Editing omega stop postion from the metadata to deal with fewer images')
        print('CHECK YOUR OMEGA EDGES AND STEP SIZE WITH np.gradient(omega_edges_deg)')
        stop = stop - (stop/num_frames_anticipated) * \
            (num_frames_anticipated-steps)
    scan = meta_df["SCAN_N"].to_numpy()
    # Find the omega start positions for each frame
    lines = [np.linspace(a, b - (b - a) / c, c)
             for a, b, c in zip(start, stop, steps)]
    omes = np.hstack([x for y, x in sorted(zip(scan, lines))])
    omega_edges_deg = np.append(omes, stop[np.shape(stop)[0]-1])
    return omes, omega_edges_deg


def _load_images(filenames, image_shape, image_dtype, start=0, stop=0):
    # How many images?
    n_imgs = stop - start
    # Generate the blank image stack
    raw_image_stack = np.zeros(
        [n_imgs, image_shape[0], image_shape[1]], image_dtype)
    for img in np.arange(n_imgs):
        raw_image_stack[img, :, :] = skimage.io.imread(filenames[start+img])
    # Return the image stack
    return raw_image_stack, start, stop


def save_image_stack(cfg, image_stack, omega_edges_deg):
    analysis_name = cfg.analysis_name
    output_directory = cfg.output_directory
    filename = output_directory + os.sep + \
        analysis_name + '_packaged_images_and_omegas.npz'
    np.savez_compressed(filename, image_stack=image_stack,
                        omega_edges_deg=omega_edges_deg)

    print(f'Data saved to: {filename}')
