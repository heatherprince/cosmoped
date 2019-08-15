import h5py
# HDF5 groups are like python dictionaries, datasets are like numpy arrays
# https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/#basic-saving-and-reading-data

filename='../chains/LCDM_all_ell.h5'
with h5py.File(filename, 'r') as f:
    print(list(f.keys()))
    print(list(f['mcmc'].keys()))
    accepted = f['mcmc']['accepted'][:]
    chain = f['mcmc']['chain'][:]
    log_prob = f['mcmc']['log_prob'][:]

import IPython; IPython.embed()
