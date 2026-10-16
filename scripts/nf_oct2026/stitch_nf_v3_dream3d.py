#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:04:28 2023

@author: lim37
"""
# %%
import argparse
import numpy as np
import os

from hexrd.transforms import xfcapi
from hexrd import rotations as rot
from hexrd.hedm import nf_config
from hexrd.material import Material
from hexrd.valunits import valWUnit
import h5py


# %%

def generate_ori_map(grain_map, exp_maps, mat, id_remap=None):
    n_grains = len(exp_maps)

    rgb_image = np.zeros(
        [grain_map.shape[0], grain_map.shape[1], grain_map.shape[2], 3], dtype='float32')
    ori_map = np.zeros(
        [grain_map.shape[0], grain_map.shape[1], grain_map.shape[2], 4], dtype='float32')
    # rgb_image[:,:,:,3] = 1.

    for ii in np.arange(n_grains):
        if id_remap is not None:
            this_grain = np.where(np.squeeze(grain_map) == id_remap[ii])
        else:
            this_grain = np.where(np.squeeze(grain_map) == ii)
        if this_grain[0].size > 0:

            ori = exp_maps[ii, :]

            rmats = xfcapi.make_rmat_of_expmap(ori)
            # print(rmats.shape)
            quats = rot.quatOfRotMat(rmats.T).T
            # quat = np.vstack([quats[1:], quats[0]])

            rgb = mat.unitcell.color_orientations(
                rmats, ref_dir=np.array([0., 1., 0.]))

            ori_map[this_grain[0], this_grain[1],
                    this_grain[2], :] = quats

            # ori_map[this_grain[0], this_grain[1],
            #           this_grain[2], 1] = rgb[0][1]
            # ori_map[this_grain[0], this_grain[1],
            #           this_grain[2], 2] = rgb[0][2]

            # color mapping
            rgb_image[this_grain[0], this_grain[1],
                      this_grain[2], 0] = rgb[0][0]
            rgb_image[this_grain[0], this_grain[1],
                      this_grain[2], 1] = rgb[0][1]
            rgb_image[this_grain[0], this_grain[1],
                      this_grain[2], 2] = rgb[0][2]

    return rgb_image, ori_map


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


def save_nf_data_for_paraview(file_dir, file_stem, grain_map, confidence_map, Xs, Ys, Zs, ori_list, mat, strain=None, tomo_mask=None, id_remap=None):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

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

    if strain is not None:
        # for i in range(len(strain)):
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map, 0], [1, 0, 2]), [2, 1, 0]), 'strain_xx')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map, 1], [1, 0, 2]), [2, 1, 0]), 'strain_yy')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map, 2], [1, 0, 2]), [2, 1, 0]), 'strain_zz')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map, 3], [1, 0, 2]), [2, 1, 0]), 'strain_yz')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map, 4], [1, 0, 2]), [2, 1, 0]), 'strain_xz')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map, 5], [1, 0, 2]), [2, 1, 0]), 'strain_xy')

    if tomo_mask is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(tomo_mask, [1, 0, 2]), [2, 1, 0]), 'tomo_mask')
    # From unitcel the color is in hsl format
    rgb_image, ori_map = generate_ori_map(grain_map, ori_list, mat, id_remap)
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(rgb_image, [1, 0, 2, 3]), [2, 1, 0, 3]), 'IPF_010')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(ori_map, [1, 0, 2, 3]), [2, 1, 0, 3]), 'quats')
    print('Writing XDMF...')
    xmdf_writer(file_dir, file_stem + '_grain_map_data')
    print('All done writing.')


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
            # return
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
        elif dims[i, 3] > 1:
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


def stitch_nf_diffraction_volumes(paths, offsets, overlap=0, confidence_threshold=0.22):
    '''
    This function stiches multiple NF diffraction volumes:
    Inputs:
        paths: .npz file locations
            size: length number of diffraction volumes
        offsets: separation of diffraction volumes
            size: length number of diffraction volumes
            These are the motor positions, thus they are likely inverted such that offset[0] will be the,
            top most diffraction volume; however, the actual value will be the smallest motor position
        overlap: overlap of diffraction volumes
            size: single valued, in voxels along stacking direction
            example: if you have 10 micron overlap, and your voxel size is 5 micron, overlap=2
        confidence_threshold: voxels below this confidence are set to 0 (no grain)
    Returns grain ids shifted by +1 so that 0 means no grain
    Assumptions:
        - stacking direction will be the shortest of the three normal directions
        - merging of overlap can be done by a simple confidence check
        - grain maps have the same dimensions
    '''
    print('Loading data.')
    # Some data lists initialization
    exp_map_list = []
    grain_map_list = []
    conf_map_list = []
    Xs_list = []
    Ys_list = []
    Zs_list = []
    nf_to_ff_id_map = []
    # if use_mask == 1:
    #     masks = []
    # Load data into lists
    for i, p in enumerate(paths):
        nf_recon = np.load(p)
        grain_map_list.append(np.flip(nf_recon['grain_map'], 0))
        conf_map_list.append(np.flip(nf_recon['confidence_map'], 0))
        exp_map_list.append(nf_recon['ori_list'])
        Xs_list.append(np.flip(nf_recon['Xs'], 0))
        Ys_list.append(np.flip(nf_recon['Ys'], 0) - offsets[i])
        Zs_list.append(np.flip(nf_recon['Zs'], 0))
        nf_to_ff_id_map.append(nf_recon['id_remap'])
        # masks.append(nf_recon['tomo_mask'])
    # What are the dimensions of our arrays?
    # Since numpy dimensions are strange this is Y X Z
    dims = np.shape(grain_map_list[0])
    print(paths)
    print(dims)
    # Initialize the merged data arrays
    num_vols = len(paths)
    grain_map_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2]), dtype=np.int32)
    confidence_map_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2]), dtype=np.float32)
    Xs_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                       dims[1], dims[2]), dtype=np.float32)
    Ys_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                       dims[1], dims[2]), dtype=np.float32)
    Zs_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                       dims[1], dims[2]), dtype=np.float32)
    vertical_position_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2]), dtype=np.float32)

    # if use_mask == 1:
    #     mask_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int8)
    print('Data Loaded.')
    print('Merging diffraction volumes.')
    # Run the merge
    if overlap != 0:
        # There is an overlap, use confidence to define which voxel to pull at each overlap
        # First assume there is no overlap
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop, :,
                           :] = grain_map_list[vol][overlap:dims[0]-overlap, :, :]
            confidence_map_full[start:stop, :,
                                :] = conf_map_list[vol][overlap:dims[0]-overlap, :, :]
            Xs_full[start:stop, :, :] = Xs_list[vol][overlap:dims[0]-overlap, :, :]
            Ys_full[start:stop, :, :] = Ys_list[vol][overlap:dims[0]-overlap, :, :]
            Zs_full[start:stop, :, :] = Zs_list[vol][overlap:dims[0]-overlap, :, :]
            vertical_position_full[start:stop, :, :] = offsets[vol]
            # if use_mask == 1:
            #     mask_full[start:stop,:,:] = masks[vol][overlap:dims[0]-overlap,:,:]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2
            print(Ys_list[vol][overlap:dims[0]-overlap, :, :].min())
            print(Ys_list[vol][overlap:dims[0]-overlap, :, :].max())
        # Now handle the overlap regions
        # Where are they?  They are the overlap voxels on either side of the volume division,
        # where the division-overlap region has to be checked against the first overlap voxels
        # of the next region, and a similar number of voxels into that region need to be checked
        # against the final voxels of the first
        # We have number of diffraction volumes - 1 overlap chunks
        # This is actually the first index of the next diffraction volume
        division_line = dims[0] - overlap*2
        for overlap_region in np.arange(num_vols-1):
            # Handle one side
            # This will be true at voxels which need replacing
            replace_voxels = np.signbit(
                confidence_map_full[division_line-overlap:division_line, :, :]-conf_map_list[overlap_region+1][0:overlap, :, :])
            grain_map_full[division_line-overlap:division_line, :,
                           :][replace_voxels] = grain_map_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            confidence_map_full[division_line-overlap:division_line, :,
                                :][replace_voxels] = conf_map_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            Xs_full[division_line-overlap:division_line, :,
                    :][replace_voxels] = Xs_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            Ys_full[division_line-overlap:division_line, :,
                    :][replace_voxels] = Ys_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            Zs_full[division_line-overlap:division_line, :,
                    :][replace_voxels] = Zs_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            vertical_position_full[division_line-overlap:division_line,
                                   :, :][replace_voxels] = offsets[overlap_region+1]
            # if use_mask == 1:
            #     mask_full[division_line-overlap:division_line,:,:][replace_voxels] = masks[overlap_region+1][0:overlap,:,:][replace_voxels]
            division_line = division_line + dims[0] - overlap*2
    else:
        # There is no overlap, scans will be stacked directly
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop, :,
                           :] = grain_map_list[vol][overlap:dims[0]-overlap, :, :]
            confidence_map_full[start:stop, :,
                                :] = conf_map_list[vol][overlap:dims[0]-overlap, :, :]
            Xs_full[start:stop, :, :] = Xs_list[vol][overlap:dims[0]-overlap, :, :]
            Ys_full[start:stop, :, :] = Ys_list[vol][overlap:dims[0]-overlap, :, :]
            Zs_full[start:stop, :, :] = Zs_list[vol][overlap:dims[0]-overlap, :, :]
            vertical_position_full[start:stop, :, :] = offsets[vol]
            # if use_mask == 1:
            #     mask_full[start:stop,:,:] = masks[vol][overlap:dims[0]-overlap,:,:]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2
    print('Diffraction volumes merged.')
    grain_map_full += 1
    grain_map_full[confidence_map_full < confidence_threshold] = 0
    return grain_map_full, confidence_map_full, Xs_full, Ys_full, Zs_full


def align_tomo_mask(tomo_mask_file, grain_map_full, Ys_full, voxel_size):
    '''
    Load the tomography mask in the orientation of the stitched volume and check that it
    sits at the same vertical positions.
    Layers are ordered by motor position, not by position in the sample, so the stitched
    volume may run in the opposite direction to the tomography mask along the first axis.
    '''
    tomo_data = np.load(tomo_mask_file)
    tomo_mask = tomo_data['mask']
    tomo_Ys = tomo_data['Ys']
    stitched_runs_down = Ys_full[0, 0, 0] > Ys_full[-1, 0, 0]
    mask_runs_down = tomo_Ys[0, 0, 0] > tomo_Ys[-1, 0, 0]
    if stitched_runs_down != mask_runs_down:
        tomo_mask = np.flip(tomo_mask, 0)
        tomo_Ys = np.flip(tomo_Ys, 0)
    if tomo_mask.shape != grain_map_full.shape:
        print(f'WARNING: tomography mask shape {tomo_mask.shape} does not match the stitched '
              f'volume {grain_map_full.shape}; is this the same mask the layers were reconstructed with?')
    else:
        y_error = np.abs(tomo_Ys[:, 0, 0] - Ys_full[:, 0, 0]).max()
        if y_error > voxel_size / 2:
            print(f'WARNING: stitched Y differs from the tomography Y by up to {y_error:.4f} mm; '
                  'check layer_center_offset against each layer vertical_motor_position')
        else:
            print('Tomography mask is aligned with the stitched volume.')
    return tomo_mask
# %%


parser = argparse.ArgumentParser(description='Stitch the NF layers into a single volume')
parser.add_argument('input_file', nargs='?', default='stitch_config.yml',
                    help='Input file for stitching the NF layers')
args = parser.parse_args()
configuration = nf_config.open_file(args.input_file)[0]
stitch = configuration.stitch

# Material
mat_file = configuration.input_files.materials_file
if mat_file is not None and not os.path.isfile(mat_file):
    mat_file = os.path.join(configuration.main_directory, mat_file)
mat = Material(name=configuration.experiment.material_name, material_file=mat_file, dmin=valWUnit(
    'lp', 'length',  0.05, 'nm'), kev=valWUnit('kev', 'energy', configuration.experiment.beam_energy, 'keV'))

fdir = configuration.output_directory
output_stem = stitch.output_stem
grains_file = configuration.input_files.grains_out_file
fnames = stitch.layer_files
vox_size = stitch.voxel_spacing
overlap_vox = stitch.overlap_voxels
tomo_mask_file = stitch.tomography_mask
# Layers are listed in order of motor position, not position in the sample.
# Each layer's saved Ys are relative to its own motor position, so these are subtracted
# to put every layer back in the tomography frame
layer_center_offset = np.array(stitch.layer_motor_positions)
print(f'Stitching {stitch.nlayers} layers with motor positions: {layer_center_offset}')
# %%

grain_map_full, confidence_map_full, Xs_full, Ys_full, Zs_full = stitch_nf_diffraction_volumes(
    fnames, layer_center_offset, overlap=overlap_vox,
    confidence_threshold=stitch.confidence_threshold)

grains = np.loadtxt(grains_file)
strain = grains[:, -6:]
ori_list = grains[:, 3:6]

tomo_mask = None
if tomo_mask_file is not None:
    tomo_mask = align_tomo_mask(tomo_mask_file, grain_map_full, Ys_full, vox_size)

# Y_vals = np.arange(-0.2075, 0.21, 0.005)
# Ys = np.tile(Y_vals, [vox_width, vox_width, 1])
# Ys_full = np.swapaxes(Ys, 0, 2)

# save_nf_data_for_paraview(fdir, 's26_full_test', grain_map_full,
# confidence_map_full, Xs_full, Ys_full, Zs_full, ori_list, mat, tomo_mask=tomo_mask)  # np.swapaxes(np.swapaxes(tomo_mask, 0, 1), 1, 2))

# save_nf_data_for_paraview(fdir, 's26_full_v4', grain_map_full,
#                           confidence_map_full, Xs_full, Ys_full, Zs_full, ori_list, mat, tomo_mask=np.swapaxes(tomo_mask, 0, 2))
save_nf_data_for_paraview(fdir, output_stem, grain_map_full,
                          # , strain=strain)
                          confidence_map_full, Xs_full, Ys_full, Zs_full, ori_list, mat, tomo_mask=tomo_mask,
                          id_remap=np.arange(1, len(ori_list) + 1))  # grain ids are index + 1; 0 is no grain

# %%
