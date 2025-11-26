import os
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sdf_nmpc.utils.df_computer import DfComputer
from sdf_nmpc.utils.data import test_dataset_from_h5

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _paths import path_to_data, path_to_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='device', nargs='?', default='cuda', help='Device to use.')
    parser.add_argument('--gt', dest='gt', action='store_true', default=False, help='Plot grount truth df.')
    args = parser.parse_args()
    device = parser.parse_args().device
    print('using', device)

    ## parameters
    path_to_data = '/home/sedith/work/colpred_ws/collision_data'
    dataset = '_virtual_depth_full.hdf5'
    vae_file = 'vae_64.pt'
    df_file = 'sdf_90.pt'
    dmax = 5

    random_skips = False
    downsampling = 1  # downsampling wrt base image resolution
    level_tol = 1e-2  # tolerance threshold for 0-lvlset
    levelset_val = 0

    ## data
    dataloader, metadata = test_dataset_from_h5(path_to_data, dataset, dmax, vae=False, device=device)

    shape_imgs = metadata['shape_imgs']
    dmax = metadata['dmax']
    hfov = metadata['hfov']
    vfov = metadata['vfov']
    is_depth = metadata['is_depth']

    ## nn
    df = torch.jit.load(os.path.join(path_to_weights, df_file))
    df.to(device)
    df.eval()

    vae = torch.jit.load(os.path.join(path_to_weights, vae_file))
    vae.to(device)
    vae.eval()

    ## generate array of bearing vectors
    y = (torch.arange(0, shape_imgs[-1], downsampling, device=device) - shape_imgs[-1]/2 + downsampling/2 + 0.5) / (shape_imgs[-1] / 2 / np.tan(hfov))
    z = (torch.arange(0, shape_imgs[-2], downsampling, device=device) - shape_imgs[-2]/2 + downsampling/2 + 0.5) / (shape_imgs[-2] / 2 / np.tan(vfov))
    grid_y, grid_z = torch.meshgrid(y, z, indexing='xy')
    rays = torch.stack([torch.ones_like(grid_y.flatten()), -grid_y.flatten(), -grid_z.flatten()], dim=1)
    rays /= rays.norm(dim=-1, keepdim=True)

    ## test
    df_cpt = DfComputer(True, dmax, hfov, vfov, df.max_df, is_depth=is_depth, device=device)

    fig = None
    for idx, (img, _) in enumerate(dataloader):
        if random_skips and np.random.rand() > 0.1:
            continue

        if not fig or not plt.fignum_exists(fig.number):
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,dmax), cmap='viridis'), ax=ax[0,1], fraction=0.046, pad=0.04)
            fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-dmax,dmax), cmap='seismic'), ax=ax[1,1], fraction=0.046, pad=0.04)
        fig.suptitle(f'test image number {idx}')

        with torch.no_grad():
            latent = vae.encoder(img)
            img_recons = vae.decoder(latent)

            ax[0,0].set_title('input image')
            ax[0,0].axis('off')
            ax[0,0].imshow(img[0,0].cpu()*dmax, vmin=0., vmax=dmax)
            ax[1,0].set_title('VAE reconstructed image')
            ax[1,0].axis('off')
            ax[1,0].imshow(img_recons[0,0].cpu()*dmax, vmin=0., vmax=dmax)

            range = torch.zeros_like(grid_y.flatten())
            not_ok = torch.ones_like(range).to(torch.bool)
            while not_ok.any():
                # print('remaining pixels:', torch.count_nonzero(not_ok).item(), end='      \r')
                points = rays[not_ok,:] * range[not_ok, None]
                if args.gt:
                    sdf_pred, _ = df_cpt.get_df(img[:,0], points)
                else:
                    sdf_pred = df(torch.hstack([points, latent.repeat(int(points.shape[0]/latent.shape[0]), 1)])).flatten()
                not_ok_old = not_ok.clone()
                not_ok_new = (sdf_pred - levelset_val).abs() > level_tol
                not_ok[not_ok_old] = not_ok_new
                range[not_ok] += sdf_pred[not_ok_new] - levelset_val
            # print('\ndone!')

            if is_depth:
                df_img = (rays[:,0] * range).reshape((int(shape_imgs[-2]/downsampling),-1)).unsqueeze(0).unsqueeze(0)
            else:
                df_img = range.reshape((int(shape_imgs[-2]/downsampling),-1)).unsqueeze(0).unsqueeze(0)
            gt_img = img
            if downsampling > 1:
                df_img = torch.nn.functional.interpolate(df_img, scale_factor=downsampling, mode='bilinear')
                # gt_img = -torch.nn.functional.max_pool2d(-img, kernel_size=downsampling)
            error_img = (df_img - gt_img*dmax) * (gt_img != 0)

            ax[0,1].set_title('SDF 0-lvl set')
            ax[0,1].axis('off')
            ax[0,1].imshow(df_img[0,0].cpu(), vmin=0., vmax=dmax)
            ax[1,1].set_title('pixel errors')
            ax[1,1].axis('off')
            ax[1,1].imshow(error_img[0,0].cpu(), vmin=-dmax, vmax=dmax, cmap='seismic')

            plt.show(block=False)
            plt.waitforbuttonpress()
