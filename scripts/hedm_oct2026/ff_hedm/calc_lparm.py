# %%
import numpy as np
from matplotlib import pyplot as plt
import copy
import os


def expand_fit(p, Xs):
    n_poly = len(p)
    Ys = np.zeros(len(Xs))
    for i, parm in enumerate(p):
        factor = parm * Xs**(n_poly-i-1)
        print(n_poly-i)
        Ys = Ys + factor
    return Ys


def write_grains_files(output_dir, analysis_stem, grain_data):
    ngrains = len(grain_data)

    f = open(
        os.path.join(
            output_dir,
            analysis_stem+'_grains.out',
        ),
        "w",
    )

    header_items = (
        "grain ID",
        "completeness",
        "chi2",
        "xi[0]",
        "xi[1]",
        "xi[2]",
        "tVec_c[0]",
        "tVec_c[1]",
        "tVec_c[2]",
        "vInv_s[0,0]",
        "vInv_s[1,1]",
        "vInv_s[2,2]",
        "vInv_s[1,2]*sqrt(2)",
        "vInv_s[0,2]*sqrt(2)",
        "vInv_s[0,1]*sqrt(2)",
        "ln(V[0,0])",
        "ln(V[1,1])",
        "ln(V[2,2])",
        "ln(V[1,2])",
        "ln(V[0,2])",
        "ln(V[0,1])",
    )
    len_items = []
    for i in header_items[1:]:
        temp = len(i)
        len_items.append(temp if temp > 19 else 19)  # for %19.12g
    fmtstr = "#%13s  " + "  ".join(["%%%ds" % i for i in len_items]) + "\n"
    f.write(fmtstr % header_items)
    for i in np.arange(grain_data.shape[0]):
        res_items = (
            grain_data[i, 0],
            grain_data[i, 1],
            grain_data[i, 2],
            grain_data[i, 3],
            grain_data[i, 4],
            grain_data[i, 5],
            grain_data[i, 6],
            grain_data[i, 7],
            grain_data[i, 8],
            grain_data[i, 9],
            grain_data[i, 10],
            grain_data[i, 11],
            grain_data[i, 12],
            grain_data[i, 13],
            grain_data[i, 14],
            grain_data[i, 15],
            grain_data[i, 16],
            grain_data[i, 17],
            grain_data[i, 18],
            grain_data[i, 19],
            grain_data[i, 20],
        )
        fmtstr = "%14d  " + \
            "  ".join(["%%%d.12g" % i for i in len_items]) + "\n"
        f.write(fmtstr % res_items)

    f.close()
    return


# %%
# fname = 'diamond-1a/FF/diamond-1a_S000_master.out'
state_num = 0
layer_num = 3

analysis_stem = f'diamond-1a_S{state_num}_L{layer_num}_fixed'

output_dir = '/usr/workspace/trimet/CHESS_2026-09/diamond-1a/FF/'

fname = f'diamond-1a/FF/diamond-1a_S{state_num}_L{layer_num}/grains_crop.out'

L0 = 3.5916

data = np.loadtxt(fname)

good_grains = np.logical_and(data[:, 2] < 0.01, data[:, 1] > 0.85)
gdata = data[good_grains, :]

comp = gdata[:, 1]
chi2 = gdata[:, 2]
COMs = gdata[:, 6:9]
strain = gdata[:, -6:]
hstrain = np.mean(strain[:, :3], axis=1)

p = np.polyfit(COMs[:, 1], hstrain, 5)

plt.plot(COMs[:, 1], hstrain, '.')
plt.plot(COMs[:, 1], strain[:, 0], '.')
plt.plot(COMs[:, 1], strain[:, 1], '.')
plt.plot(COMs[:, 1], strain[:, 2], '.')

Xs = np.linspace(-0.2, 0.115, 100)
# Ys = p[0]*Xs**5 + p[1]*Xs**4 + p[2]*Xs**3 + p[3]*Xs**2 + p[4]*Xs+p[5]
Ys = expand_fit(p, Xs)
plt.plot(Xs, Ys, 'o')
# %%

all_COMs = data[:, 6:9]
all_strain = data[:, -6:]

Xs = all_COMs[:, 1]
Ys = expand_fit(p, Xs)
plt.plot(Xs, Ys, '.')

fixed = Ys


# plt.plot(Xs, hstrain-fixed, 'o')
strain_fixed = copy.deepcopy(all_strain)
strain_fixed[:, 0] = strain_fixed[:, 0]-fixed/3
strain_fixed[:, 1] = strain_fixed[:, 1]-fixed/3
strain_fixed[:, 2] = strain_fixed[:, 2]-fixed/3

plt.ylim(-2e-3, 2e-3)


avg_hstrain = np.mean(hstrain[np.abs(hstrain) < 1e-2])

fixed_gdata = copy.deepcopy(data)
fixed_gdata[:, -6:] = strain_fixed
# %%
L = avg_hstrain*L0+L0
print(avg_hstrain)
print(L)
# %%

write_grains_files(output_dir, analysis_stem,
                   fixed_gdata)

# %%
