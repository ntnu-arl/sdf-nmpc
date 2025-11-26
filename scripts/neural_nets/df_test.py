import os
import argparse
import numpy as np
import torch
from torchinfo import summary
import matplotlib as mpl
import matplotlib.pyplot as plt
from sdf_nmpc.utils.df_computer import DfComputer
from sdf_nmpc.utils.pos_sampler import PosSampler
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
    dataset = '_virtual_depth_sim.hdf5'
    vae_file = 'vae_64.pt'
    sdf_file = 'sdf/2025-05-22T21:29:31/weights.pt'
    dmax = 5

    angles = np.array([20, 10, 0, -10, -20])  # [degrees], elevation angles at which slices of SDF is plotted
    fcontour_levels = 20  # nb of displayed levels in fcontour
    lvlset = 0.35  # level-set to display
    lvl_tol = 0.01  # tolerance threshold for lvlset
    nb_points = 10000
    shuffle = False

    ## data
    dataloader, metadata = test_dataset_from_h5(path_to_data, dataset, dmax, shuffle=shuffle, vae=False, device=device)

    shape_imgs = metadata['shape_imgs']
    hfov = metadata['hfov']
    vfov = metadata['vfov']
    is_depth = metadata['is_depth']
    is_spherical = metadata['is_spherical']

    ## nn
    df = torch.jit.load(os.path.join(path_to_weights, sdf_file))
    df.to(device)
    df.eval()

    vae = torch.jit.load(os.path.join(path_to_weights, vae_file))
    vae.to(device)
    vae.eval()

    ## summary
    summary(df, input_size=[(1, 3 + vae.size_latent)], device=device, depth=6)

    ## test
    margin = 20
    df_cpt = DfComputer(df.signed, dmax, hfov, vfov, df.max_df, batch_size=1500, is_depth=is_depth, is_spherical=is_spherical, device=device)
    pos_sampler = PosSampler(dmax, hfov, vfov, 40, device=device)

    fov_h = dmax * (np.tan(hfov)) / np.sqrt(1+np.tan(hfov)**2)
    fov_d = dmax / np.sqrt(1+np.tan(hfov)**2) * (-1 if hfov > np.pi/2 else 1)
    dsup = dmax * (100+margin)/100
    dinf = -0.2 if hfov < np.pi / 2 else -dsup
    fig = None
    for idx, (img_in, img_out) in enumerate(dataloader):
        if not fig or not plt.fignum_exists(fig.number):
            fig, ax = plt.subplots(nrows=len(angles), ncols=4)
            cbar = fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(df_cpt.min_df, df_cpt.max_df), cmap='magma'), ax=ax[:,-2:], fraction=0.15, pad=0.04)
            cbar.ax.axhline(y=lvlset, color='blue', linewidth=2)
            cbar.ax.axhline(y=0, color='white', linewidth=2)
            for j, _ in enumerate(angles):
                ax[j,0].axis('off')
                ax[j,1].axis('off')
                ax[j,2].set_xlim([-dsup, dsup])
                ax[j,2].set_ylim([dinf, dsup])
                ax[j,2].invert_xaxis()
                ax[j,3].set_xlim([-dsup, dsup])
                ax[j,3].set_ylim([dinf, dsup])
                ax[j,3].invert_xaxis()
        fig.suptitle(f'test image number {idx}')

        with torch.no_grad():
            latent = vae.encoder(img_in)
            img_recons = vae.decoder(latent)
        ax[0,0].imshow(img_recons[0,0].cpu()*dmax, vmin=0., vmax=dmax, cmap='magma')

        for j, bearing_v_deg in enumerate(angles):
            bearing_v_cos = np.cos(bearing_v_deg * np.pi / 180)
            pixel_v = np.clip(-bearing_v_deg*np.pi/180/vfov * shape_imgs[1]/2 + shape_imgs[1]/2, 0, shape_imgs[1]-1)
            with torch.no_grad():
                points = pos_sampler.grid_frustrum_slice(nb_points, bearing_v_deg, True)
                df_gt, _ = df_cpt.get_df(img_out[:,0], points)
                df_pred = df(torch.hstack([points, latent.repeat(int(points.shape[0]/latent.shape[0]), 1)]))

            df_pred = df_pred.cpu()
            df_gt = df_gt.cpu()

            ax[j,1].imshow(img_out[0,0].cpu()*dmax, vmin=0., vmax=dmax, cmap='magma')
            ax[j,1].hlines(pixel_v, 0, shape_imgs[-1], color='green')

            shape_grid = (int(points.shape[0]**(1/2)),int(points.shape[0]**(1/2)))
            x = points[:,0].cpu().reshape(shape_grid).T
            y = points[:,1].cpu().reshape(shape_grid).T

            ax[j,2].contourf(y, x, df_gt.reshape(shape_grid).T, levels=fcontour_levels, vmin=df_cpt.min_df, vmax=df_cpt.max_df, cmap='magma')
            ax[j,2].contourf(y, x, df_pred.reshape(shape_grid).T, levels=[lvlset-lvl_tol,lvlset+lvl_tol], colors='blue', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,2].contourf(y, x, df_pred.reshape(shape_grid).T, levels=[0-lvl_tol,0+lvl_tol], colors='gray', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,2].contourf(y, x, df_gt.reshape(shape_grid).T, levels=[lvlset-lvl_tol,lvlset+lvl_tol], colors='cyan', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,2].contourf(y, x, df_gt.reshape(shape_grid).T, levels=[0-lvl_tol,0+lvl_tol], colors='white', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,2].plot([0, -fov_h*bearing_v_cos], [0, fov_d*bearing_v_cos], color='green', linewidth=2)
            ax[j,2].plot([0, fov_h*bearing_v_cos], [0, fov_d*bearing_v_cos], color='green', linewidth=2)
            arc = mpl.patches.Arc((0, 0), dmax*2*bearing_v_cos, dmax*2*bearing_v_cos, color='green', linewidth=2, angle=90, theta1=-hfov/np.pi*180, theta2=hfov/np.pi*180)
            ax[j,2].add_patch(arc)

            ax[j,3].contourf(y, x, df_pred.reshape(shape_grid).T, levels=fcontour_levels, vmin=df_cpt.min_df, vmax=df_cpt.max_df, cmap='magma')
            ax[j,3].contourf(y, x, df_gt.reshape(shape_grid).T, levels=[lvlset-lvl_tol,lvlset+lvl_tol], colors='cyan', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,3].contourf(y, x, df_gt.reshape(shape_grid).T, levels=[0-lvl_tol,0+lvl_tol], colors='white', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,3].contourf(y, x, df_pred.reshape(shape_grid).T, levels=[lvlset-lvl_tol,lvlset+lvl_tol], colors='blue', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,3].contourf(y, x, df_pred.reshape(shape_grid).T, levels=[0-lvl_tol,0+lvl_tol], colors='gray', vmin=df_cpt.min_df, vmax=df_cpt.max_df)
            ax[j,3].plot([0, -fov_h*bearing_v_cos], [0, fov_d*bearing_v_cos], color='green', linewidth=2)
            ax[j,3].plot([0, fov_h*bearing_v_cos], [0, fov_d*bearing_v_cos], color='green', linewidth=2)
            arc = mpl.patches.Arc((0, 0), dmax*2*bearing_v_cos, dmax*2*bearing_v_cos, color='green', linewidth=2, angle=90, theta1=-hfov/np.pi*180, theta2=hfov/np.pi*180)
            ax[j,3].add_patch(arc)

        plt.show(block=False)
        plt.waitforbuttonpress()
