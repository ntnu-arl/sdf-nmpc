import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sdf_nmpc.utils.df_computer import DfComputer
from sdf_nmpc.utils.collision_checker import ColChecker
from sdf_nmpc.utils.pos_sampler import PosSampler
from sdf_nmpc.utils.data import test_dataset_from_h5
from sdf_nmpc.utils.visualization import Imgs2Points

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _paths import path_to_data, path_to_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='device', nargs='?', default='cuda', help='Device to use.')
    parser.add_argument('--gt', dest='gt', action='store_true', default=False, help='Plot grount truth df.')
    args = parser.parse_args()
    device = args.device
    print('using', device)

    dataset = '_virtual_depth_full.hdf5'
    vae_file = 'vae_64.pt'
    df_file = 'sdf_90.pt'
    dmax = 5
    in_frustrum = True
    step = 1
    random_skips = False

    ## data
    dataloader, metadata = test_dataset_from_h5(path_to_data, dataset, dmax, vae=False, device=device)

    shape_imgs = metadata['shape_imgs']
    hfov = metadata['hfov']
    vfov = metadata['vfov']
    is_depth = metadata['is_depth']
    is_spherical = metadata['is_spherical']

    ## nn
    df = torch.jit.load(os.path.join(path_to_weights, df_file))
    df.to(device)
    df.eval()

    vae = torch.jit.load(os.path.join(path_to_weights, vae_file))
    vae.to(device)
    vae.eval()

    ## setup
    df_cpt = DfComputer(True, dmax, hfov, vfov, df.max_df, is_depth=is_depth, is_spherical=is_spherical, device=device)
    pos_sampler = PosSampler(dmax, hfov, vfov, margin=20, device=device)
    colcheck = ColChecker(dmax, hfov, vfov, 0, is_depth=is_depth, is_spherical=is_spherical, outside='free', device=device)
    img_to_points = Imgs2Points(is_depth, is_spherical, dmax, hfov, vfov, remove_dmax=False, downsamp=5, device=device)

    grid = pos_sampler.grid_sphere_fixed_step(step, in_frustrum=in_frustrum, frustrum_is_spherical=is_spherical, add_margin=False)
    grid = grid[torch.linalg.vector_norm(grid, dim=1) < dmax*0.95]
    grid_with_grad = grid.clone().requires_grad_(True)
    grid_cpu = grid.cpu()
    print('nb points:', grid.shape[0])

    ## test
    fig = None
    for idx, (img, img_out) in enumerate(dataloader):
        if random_skips and np.random.rand() > 0.1:
            continue

        if not fig or not plt.fignum_exists(fig.number):
            fig = plt.figure()
        fig.suptitle(f'test image number {idx}')

        with torch.no_grad():
            latent = vae.encoder(img)

        ## show imgs
        ax = fig.add_subplot(131)
        ax.imshow(img[0,0].cpu()*dmax, vmin=0., vmax=dmax)

        ## show pc
        ax = fig.add_subplot(1,3,(2,3), projection='3d')
        ax.scatter3D(0,0,0, color='k', alpha=0.8, marker='s', s=10)
        pc = img_to_points(img[0]).cpu()
        ax.scatter3D(pc[:,0], pc[:,1], pc[:,2], color='k', alpha=0.2, marker='.', s=8)

        # ## occupancy
        # occ = colcheck.check_image_points(img[0], grid).cpu().numpy()
        # ax.scatter3D(grid_cpu[occ,0], grid_cpu[occ,1], grid_cpu[occ,2], color='b', alpha=0.8, marker='.', s=10)

        ## df ground truth
        df_gt, grad_gt = df_cpt.get_df(img[0], grid)
        ## neural df
        df_nn = df(torch.hstack([grid_with_grad, latent.repeat(int(grid.shape[0]/latent.shape[0]), 1)]))
        grad = torch.autograd.grad(df_nn, grid_with_grad, grad_outputs=torch.ones_like(df_nn), retain_graph=True, create_graph=True)[0]
        
        if len(df_nn.shape) > 1: df_nn = df_nn.squeeze(1)
        df_nn = df_nn.detach().cpu()
        grad = grad.detach().cpu()
        df_gt = df_gt.detach().cpu()
        grad_gt = grad_gt.detach().cpu()
        closest_points = grid_cpu - df_nn[:,None] * torch.nn.functional.normalize(grad, dim=1)
        closest_points = grid_cpu - df_nn[:,None] * grad
        closest_points_gt = grid_cpu - df_gt[:,None] * grad_gt

        # show points
        # ax.scatter3D(grid_cpu[:,0], grid_cpu[:,1], grid_cpu[:,2], color='r', vmin=-df.max_df, vmax=df.max_df, alpha=0.8, marker='o', s=40)
        ax.scatter3D(grid_cpu[:,0], grid_cpu[:,1], grid_cpu[:,2], c=torch.sign(df_gt), vmin=-df.max_df, vmax=df.max_df, alpha=0.3, marker='o', s=40)

        ## show points and their sdf
        ax.scatter3D(closest_points[:,0], closest_points[:,1], closest_points[:,2], color='b', alpha=0.5, marker='o', s=20)
        for p, c in zip(grid_cpu, closest_points):
            ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], color='b', alpha=0.5)

        ## show gt points and their sdf
        ax.scatter3D(closest_points_gt[:,0], closest_points_gt[:,1], closest_points_gt[:,2], color='g', alpha=0.8, marker='o', s=20)
        for p, c in zip(grid_cpu, closest_points_gt):
            ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], color='g', alpha=0.8)


        ## display
        plt.show()
