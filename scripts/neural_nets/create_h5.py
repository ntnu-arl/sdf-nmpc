import os
import numpy as np
import h5py

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _paths import path_to_data


path_to_data = '/home/sedith/work/colpred_ws/collision_data'
ratio_test = 0.1
is_sim = False
description = 'all real images'
out_file = '_virtual_lidar_full.hdf5'
key = 'images'
in_files = [
    'lidar_isaac.hdf5',
    'lidar_angelos.hdf5',
    'lidar_real.hdf5',
    # 'depth_isaac_90.hdf5',
    # 'depth_isaac_125.hdf5',
    # 'depth_oracle.hdf5',
    # 'depth_d455.hdf5',
    # 'tartanair.hdf5',
]

## check input files
nb_imgs = []
nb_train = []
nb_test = []
dmax = 0
for fin in in_files:
    with h5py.File(os.path.join(path_to_data, fin),'r') as h5in:
        if not nb_imgs:
            shape_imgs = h5in[key].shape[1:]
            dtype = h5in[key].dtype
            is_depth = h5in.attrs['is_depth']
            is_spherical = h5in.attrs['is_spherical']
        else:
            assert shape_imgs == h5in[key].shape[1:]
            assert dtype == h5in[key].dtype
            assert is_depth == h5in.attrs['is_depth']
            assert is_spherical == h5in.attrs['is_spherical']
        if is_sim:
            assert h5in.attrs['is_sim']
        dmax = max(dmax, h5in.attrs['dmax'])
        nb_imgs.append(h5in[key].shape[0])
        nb_test.append(int(nb_imgs[-1]*ratio_test))
        nb_train.append(nb_imgs[-1] - nb_test[-1])

virtual_sources = [h5py.VirtualSource(fin, name=key, shape=(nb,) + shape_imgs) for nb, fin in zip(nb_imgs, in_files)]

virtual_layouts = {
    'train': h5py.VirtualLayout(shape=(sum(nb_train),) + shape_imgs, dtype=dtype),
    'test': h5py.VirtualLayout(shape=(sum(nb_test),) + shape_imgs, dtype=dtype)
}


## write to virtual file
with h5py.File(os.path.join(path_to_data, out_file),'w') as h5out:
    ## write attributes from first input file
    fin = in_files[0]
    with h5py.File(os.path.join(path_to_data, fin),'r') as h5in:
        for attr in h5in.attrs.keys():
            if attr == 'is_sim':
                h5out.attrs.create(attr, is_sim)
            elif attr == 'description':
                h5out.attrs.create(attr, description)
            elif attr == 'dmax':
                h5out.attrs.create(attr, dmax)
            else:
                h5out.attrs.create(attr, h5in.attrs[attr])

    ## add new metadata attributes
    h5out.attrs.create('input_files', in_files)

    ## print all attributes
    for attr in h5out.attrs.keys():
        print('attribute', attr, h5out.attrs[attr])


    ## assign sources to layouts
    idx_train = 0
    idx_test = 0
    for vs, ntrain, ntest in zip(virtual_sources, nb_train, nb_test):
        virtual_layouts['train'][idx_train:idx_train+ntrain] = vs[:ntrain]
        virtual_layouts['test'][idx_test:idx_test+ntest] = vs[ntrain:]
        idx_train += ntrain
        idx_test += ntest

    ## create and populate virtual groups
    for group in ['train', 'test']:
        print('creating group:', group)
        h5out.create_group(group)
        h5out[group].create_virtual_dataset(key, virtual_layouts[group])
