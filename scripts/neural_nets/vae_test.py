import os
import argparse
import numpy as np
import torch
from torchinfo import summary
import matplotlib as mpl
import matplotlib.pyplot as plt
from sdf_nmpc.utils.data import test_dataset_from_h5, train_dataset_from_h5

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _paths import path_to_data, path_to_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='device', nargs='?', default='cuda', help='Device to use.')
    device = parser.parse_args().device
    print('using', device)

    ## parameters
    dataset = '_virtual_depth_real.hdf5'
    vae_file = 'vae_64.pt'
    dmax = 5
    shuffle = False

    ## data
    dataloader, metadata = test_dataset_from_h5(path_to_data, dataset, dmax, shuffle=shuffle, vae=True, col_map=True, device=device)
    shape_imgs = metadata['shape_imgs']

    ## nn
    vae = torch.jit.load(os.path.join(path_to_weights, vae_file))
    vae.to(device)
    vae.eval()

    ## test
    fig = None
    for idx, (img_in, img_out) in enumerate(dataloader):
        if not fig or not plt.fignum_exists(fig.number):
            fig = plt.figure()
        fig.suptitle(f'test image number {idx} - {vae_file}')

        with torch.no_grad():
            vae_reconst = vae(img_in)
            error_img = (vae_reconst - img_out) * (img_out != 0)

            ax = plt.subplot(221)
            ax.set_title('input image')
            ax.axis('off')
            ax.imshow(img_in[0,0,:,:].cpu()*dmax, vmin=0., vmax=dmax, cmap='magma')
            ax = plt.subplot(223)
            ax.set_title('target image')
            ax.axis('off')
            ax.imshow(img_out[0,0,:,:].cpu()*dmax, vmin=0., vmax=dmax, cmap='magma')
            ax = plt.subplot(222)
            ax.axis('off')
            ax.set_title('reconstructed image')
            ax.imshow(vae_reconst[0,0,:,:].cpu()*dmax, vmin=0., vmax=dmax, cmap='magma')
            ax = plt.subplot(224)
            ax.axis('off')
            ax.set_title('signed pixel errors')
            ax.imshow(error_img[0,0,:,:].cpu()*dmax, vmin=-1, vmax=1, cmap='seismic')
            fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1,1), cmap='seismic'), ax=ax, fraction=0.046, pad=0.04)
            plt.show(block=False)
            plt.waitforbuttonpress()
