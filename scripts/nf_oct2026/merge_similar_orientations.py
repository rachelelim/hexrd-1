# %%
import numpy as np

import logging
import os

from hexrd import instrument, constants, rotations

# %%


def merge_similar_orientations(quats: np.ndarray,  misorientation_deg: float = 0.5, sym_group: str = None) -> np.ndarray:
    """
    Merge orientations within a misorientation threshold.

    quats are crystal-to-sample orientations (4, n). If sym_group (a Laue group
    name such as 'oh') is given, symmetry-equivalent orientations count as the
    same orientation.
    """
    working_quats = quats
    merged_quats = []
    misorientation_rad = np.radians(misorientation_deg)
    symmetries = None
    if sym_group is not None:
        symmetries = (rotations.quatOfLaueGroup(sym_group),)
    # rotations.misorientation applies the symmetry operators on the sample-to-crystal side
    # so conjugate the crystal-to-sample quats to apply the crystal symmetry correctly
    conjugate = np.array([[1.], [-1.], [-1.], [-1.]])

    while working_quats.shape[1] > 0:
        ref_quat = np.atleast_2d(working_quats[:, 0]).T
        test_quats = np.atleast_2d(working_quats)
        misorientations, _ = rotations.misorientation(
            conjugate * ref_quat, conjugate * test_quats, symmetries)
        idx_to_merge = misorientations < misorientation_rad
        merged_quats.append(working_quats[:, 0])
        working_quats = np.delete(working_quats, idx_to_merge, axis=1)

    final_quats = np.stack(merged_quats, axis=1)
    final_exp_maps = rotations.expMapOfQuat(final_quats)
    print(
        f'{quats.shape[1]} grains merged to {final_quats.shape[1]} grains with {misorientation_deg} degrees misorientation')
    # logging.info(
    #     f'Found {final_quats.shape[1]} additional grains during brute force search.')
    return final_exp_maps


def save_grains_out(output_path: str, exp_maps: np.ndarray) -> None:
    """Save grains.out file with all found grains."""

    gw = instrument.GrainDataWriter(output_path)
    for gid, ori in enumerate(exp_maps):
        grain_params = np.hstack(
            [ori, constants.zeros_3, constants.identity_6x1])
        gw.dump_grain(gid, 1., 0., grain_params)
    gw.close()
    logging.info(f'Saved grains.out to {output_path}')


# %%


# 'NF/SS316L-1_S0_L3_real_grains.out'
# input_fname = '/usr/workspace/lim37/ER-disloc/voisin-4783-a/SS316L-3/twin_finding/S0_L0_wtwins_grains.out'
# NF/SS316L-1_S0_L3_merged_grains.out'
input_fname = 'brass_S0_L%d_real_grains.out'
output_fname = 'brass_S0_L%d_merged_grains.out'
for i in range(8):

    gdata = np.loadtxt(input_fname % i)
    expmaps = gdata[:, 3:6]
    quats = rotations.quatOfExpMap(expmaps.T)

    final_oris = merge_similar_orientations(quats, 0.5, 'oh')

    save_grains_out(output_fname % i, final_oris.T)

# %%
# input_fname = '../brass_S0_L%d/grains.out'
input_fname = 'brass_S0_L%d_real_grains.out'
expmap_list = []
for i in range(8):
    gdata = np.loadtxt(input_fname % i)
    expmaps = gdata[:, 3:6]
    expmap_list.append(expmaps)
# %%
all_oris = 0

for i in range(8):
    all_oris += len(expmap_list[i])
# all_oris = len(expmap_list[0])+len(expmap_list[1]) + \
#     len(expmap_list[2])+len(expmap_list[3])

# %%
full_ori_list = np.array([0, 0, 0])

for i in range(8):
    full_ori_list = np.vstack(
        [full_ori_list, expmap_list[i]])
# full_ori_list = np.vstack(
    # [expmap_list[0], expmap_list[1], expmap_list[2], expmap_list[3]])
# %%
all_quats = rotations.quatOfExpMap(full_ori_list[1:, :].T)

all_final_oris = merge_similar_orientations(all_quats, 0.5, 'oh')

# %%
save_grains_out('NF_merged_grains.out', all_final_oris.T)
# %%
