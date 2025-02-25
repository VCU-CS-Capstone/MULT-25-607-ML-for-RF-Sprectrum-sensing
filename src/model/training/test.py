import h5py
import numpy as np

gmin = 999
gmax = -999
keymax = ""
keymin = ""

with h5py.File("overair.h5", "r") as f:
    for key in f.keys():
        lmin = np.min(f[key][:][np.isfinite(f[key][:])])
        lmax = np.max(f[key][:][np.isfinite(f[key][:])])
        if lmin < gmin:
            gmin = lmin
            keymin = key
        if lmax > gmax:
            gmax = lmax
            keymax = key
    print(gmin, keymin)
    print(gmax, keymax)

gmin = 999
gmax = -999
keymax = ""
keymin = ""

with h5py.File("train.h5", "r") as f:
    for key in f.keys():
        lmin = np.min(f[key][:][np.isfinite(f[key][:])])
        lmax = np.max(f[key][:][np.isfinite(f[key][:])])
        if lmin < gmin:
            gmin = lmin
            keymin = key
        if lmax > gmax:
            gmax = lmax
            keymax = key
    print(gmin, keymin)
    print(gmax, keymax)
