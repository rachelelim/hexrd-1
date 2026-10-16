# %%
import numpy as np
from hexrd import instrument, constants, rotations
import logging


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

# layer = 7
for layer in range(8):
    fname = 'brass_S0_L%d_grain_map_data.npz' % layer
    # 'SS316L-1_S0_L%d_merged_grains.out' % layer
    input_fname = 'brass_S0_L%d_grains.out' % layer

    nf_data = np.load(fname)
    grain_map = nf_data['grain_map']
    confidence_map = nf_data['confidence_map']

    real_grains = np.unique(grain_map).astype(int)


    gdata = np.loadtxt(input_fname)
    expmaps = gdata[real_grains, 3:6]

    print(f'{len(real_grains)} real grains from {len(gdata)} grains')


    # %
    save_grains_out('brass_S0_L%d_real_grains.out' % layer, expmaps)

# %%
