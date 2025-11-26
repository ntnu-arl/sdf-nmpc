import os
import time
import datetime
import argparse
import numpy as np
import torch
from torchinfo import summary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # shuts down warning message when loading tensorboard
from torch.utils.tensorboard import SummaryWriter
from sdf_nmpc.network.neural_df import NeuralDF
from sdf_nmpc.utils.losses import loss_sdf
from sdf_nmpc.utils.df_computer import DfComputer
from sdf_nmpc.utils.pos_sampler import PosSampler
from sdf_nmpc.utils.data import train_dataset_from_h5
from sdf_nmpc.utils.layer_init import init_linear_layer_sine

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _paths import path_to_data, path_to_weights


def samples_points(pos_sampler, imgs, nb_points_frustrum, nb_points_ball, nb_points_obs, nb_points_margin, ball_size, device):
    ## sample states in all regions
    states_frustrum = pos_sampler.sample_pos_in_frustrum(imgs.shape[0]*nb_points_frustrum, add_margin=False)
    states_ball = pos_sampler.sample_pos_in_ball(imgs.shape[0]*nb_points_ball, ball_size, add_margin=False)
    states_margin = pos_sampler.sample_pos_in_frustrum_margin(imgs.shape[0]*nb_points_margin)
    states_obs = pos_sampler.sample_pos_around_obs(imgs, nb_points_obs, mode='random', std=0.1)

    ## concatenate both state arrays such that each block of points_per_img rows contains desired points in the respective ranges
    states = torch.cat([
        states_frustrum.reshape((-1, nb_points_frustrum, 3)),
        states_ball.reshape((-1, nb_points_ball, 3)),
        states_obs.reshape((-1, nb_points_obs, 3)),
        states_margin.reshape((-1, nb_points_margin, 3)),
    ], dim=1).reshape((-1, 3))

    return states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='device', nargs='?', default='cuda', help='Device to use.')
    device = parser.parse_args().device
    print('using', device)

    ## parameters
    dataset = '_virtual_depth_sim.hdf5'
    vae_file = 'vae_64.pt'
    sdf_file_prefix = 'sdf'
    max_df = 1
    dmax = 5
    signed_df = True
    train_valid_ratio = 0.9  # ratio of train and valid data
    nb_epochs = 200
    lr_start = 5e-5
    lr_min = 1e-5
    lr_nb_steps = 20
    weight_decay = 1e-5
    batch_size_train = 50  # size of train batches
    batch_size_valid = 30  # size of valid batches
    points_per_img = 2500  # nb of points sampled per image
    ratio_points_ball = 0.2  # ratio of points that are sampled of the ball around origin rather than in frustrum
    ratio_points_obs = 0.4  # ratio of points that are sampled of the ball around obstacles rather than in frustrum
    ratio_points_margin = 0.15  # ratio of points that are sampled in the frustrum margin
    assert (ratio_points_ball*points_per_img) % 1 == 0
    assert (ratio_points_obs*points_per_img) % 1 == 0
    assert (ratio_points_margin*points_per_img) % 1 == 0
    nb_points_ball = int(points_per_img*ratio_points_ball)
    nb_points_obs = int(points_per_img*ratio_points_obs)
    nb_points_margin = int(points_per_img*ratio_points_margin)
    nb_points_frustrum = points_per_img - nb_points_obs - nb_points_ball - nb_points_margin
    close_ball_size = 0.75  # size of ball for ensuring good sampling close to the camera origin
    sdf_losses_weights = (50, 0, 1 / 60, 5)
    restart_from_epoch = 0

    ## data
    (train_dataloader, valid_dataloader), metadata = train_dataset_from_h5( \
        path_to_data, dataset, dmax, batch_size_train=batch_size_train, batch_size_valid=batch_size_valid, \
        train_valid_ratio=train_valid_ratio, vae=False, device=device)

    hfov = metadata['hfov']
    vfov = metadata['vfov']
    is_depth = metadata['is_depth']
    is_spherical = metadata['is_spherical']

    ## nn
    vae = torch.jit.load(os.path.join(path_to_weights, vae_file))
    vae.to(device)
    vae.eval()

    if restart_from_epoch:
        now = '2025-05-26T13:06:24'
    else:
        now = datetime.datetime.now().isoformat()[:-7]
        # now = 'test'
    nn_file_now = os.path.join(path_to_weights, sdf_file_prefix, now)

    kwargs = {
        'signed':signed_df, 'size_latent':vae.size_latent, 'nb_freqs':5, 'res':'full',
        'embed':'oct', 'act':'sin', 'dropout_rate':0.1, 'w0':20,
    }
    kwargs_2 = [{'layer_sizes':[128,128,128,128]}, {'layer_sizes':[256,256,128,64]}]

    names = [''.join([f'_{i}' for i in list(kw.values())])[1:] for kw in kwargs_2]
    if restart_from_epoch:
        nns = [torch.jit.load(f'{nn_file_now}/{name}/epochs/e{restart_from_epoch-1}.pt') for name in names]
    else:
        nns = []
        for k2, name in zip(kwargs_2, names):
            os.makedirs(f'{nn_file_now}/{name}/epochs', exist_ok=True)
            nn = NeuralDF(**k2, **kwargs)
            init_linear_layer_sine(nn.layers, nn.w0)
            nn = torch.jit.script(nn)
            nns.append(nn)

    ## nn summary
    for name, nn in zip(names, nns):
        nn.to(device)
        nn_sum = summary(nn, input_size=[(1, 3 + vae.size_latent)], device=device, verbose=0)
        print(f'Training model {name}: {nn_sum.trainable_params} parameters')

    ## sampler, checker...
    df_cpt = DfComputer(signed_df, dmax, hfov, vfov, max_df, is_depth=is_depth, is_spherical=is_spherical, batch_size=5000, device=device)
    pos_sampler = PosSampler(dmax, hfov, vfov, margin=40, is_spherical=is_spherical, device=device)

    ## tensorboard writers
    tsb_train = [SummaryWriter(log_dir=os.path.join(nn_file_now, name, 'train')) for name in names]
    tsb_valid = [SummaryWriter(log_dir=os.path.join(nn_file_now, name, 'valid')) for name in names]

    ## training printing parameters
    nb_train_batches = len(train_dataloader)
    nb_batches_print_train = int(nb_train_batches/5)
    nb_valid_batches = len(valid_dataloader)
    nb_batches_print_valid = int(nb_valid_batches/2)

    ## training
    optimizers = [
        torch.optim.AdamW(nn.parameters(), lr=lr_start, weight_decay=weight_decay)
        for nn in nns
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=lr_nb_steps, eta_min=lr_min)
        for optim in optimizers
    ]
    ## step scheduler if restarting from epoch
    for scheduler in schedulers:
        for _ in range(min(restart_from_epoch, lr_nb_steps)):
            scheduler.step()

    for idx_epoch in range(restart_from_epoch, nb_epochs):
        tic = time.time()
        print(f'-------------------------------\nepoch {idx_epoch} -- lr: {schedulers[0].get_last_lr()[0]:.2e}')
        losses_train = np.array([[0. for _ in sdf_losses_weights] for _ in nns])  # aggregates losses across batches
        losses_valid = np.array([[0. for _ in sdf_losses_weights] for _ in nns])
        disp_text = ['' for _ in nns]  # fstrings for display in terminal after batch

        ## train
        for nn in nns: nn.train()
        for idx_batch, (imgs_in, imgs_out) in enumerate(train_dataloader):
            for opt in optimizers: opt.zero_grad()

            with torch.no_grad():
                mean, logvar = vae.encoder.mean_logvar(imgs_in)
                # latents = vae.encoder.sample(mean, logvar).repeat_interleave(points_per_img, 0)
                # latents = mean.repeat_interleave(points_per_img, 0)
                latents = vae.encoder.sample(mean, logvar, points_per_img)
                states = samples_points(pos_sampler, imgs_out, nb_points_frustrum, nb_points_ball, nb_points_obs, nb_points_margin, close_ball_size, device)
                df_gt, df_grads = df_cpt.get_df(imgs_out[:,0], states)

            states.requires_grad_(True)  # required for computing gradient loss

            ## train all networks
            disp_text = ['' for _ in nns]  # fstrings for display in terminal after batch
            for i, nn in enumerate(nns):
                sdf_nn = nn(torch.hstack([states, latents]))
                losses = loss_sdf(sdf_nn, states, df_grads, df_gt)

                ## backpropagation
                loss = sum([w * l for l,w in zip(losses, sdf_losses_weights)])
                loss.backward()
                optimizers[i].step()

                for j,l in enumerate(losses):
                    losses_train[i,j] += l.item()

                disp_text[i] =  f' - {names[i]} {losses[0].item():>.4f} / {losses[1].item():>.4f} / {losses[2].item():>.4f} / {losses[3].item():>.4f}'

            ## print
            print(f'train batch: {idx_batch+1}/{nb_train_batches}' + \
                  disp_text[0] + \
                  f' - elapsed time: {(time.time()-tic):.3f}', end='\r')
            if not ((idx_batch+1) % nb_batches_print_train): print()

        ## accounting
        losses_train /= nb_train_batches
        for i, _ in enumerate(names):
            tsb_train[i].add_scalar(f'loss/regression', losses_train[i,0], idx_epoch)
            tsb_train[i].add_scalar(f'loss/gradient', losses_train[i,1], idx_epoch)
            tsb_train[i].add_scalar(f'loss/gradient_dir', losses_train[i,2], idx_epoch)
            tsb_train[i].add_scalar(f'loss/eikonal', losses_train[i,3], idx_epoch)
            tsb_train[i].add_scalar(f'loss/total', losses_train[i,:].sum(axis=-1), idx_epoch)

        ## valid
        for nn in nns: nn.eval()
        for idx_batch, (imgs_in, imgs_out) in enumerate(valid_dataloader):
            with torch.no_grad():
                latents = vae.encoder(imgs_in).repeat_interleave(points_per_img, 0)
                states = samples_points(pos_sampler, imgs_out, nb_points_frustrum, nb_points_ball, nb_points_obs, nb_points_margin, close_ball_size, device)
                df_gt, df_grads = df_cpt.get_df(imgs_out[:,0], states)

            states.requires_grad_(True)  # required for computing gradient loss

            ## validate all networks
            for i, nn in enumerate(nns):
                sdf_nn = nn(torch.hstack([states, latents]))
                losses = loss_sdf(sdf_nn, states, df_grads, df_gt)

                for j,l in enumerate(losses):
                    losses_valid[i,j] += l.item()

                disp_text[i] =  f' - {names[i]} {losses[0].item():>.4f} / {losses[1].item():>.4f} / {losses[2].item():>.4f} / {losses[3].item():>.4f}'

            ## print
            print(f'valid batch: {idx_batch+1}/{nb_valid_batches}' + \
                  disp_text[0] + \
                  f' - elapsed time: {time.time()-tic:.3f}', end='\r')
            if not ((idx_batch+1) % nb_batches_print_valid): print()

        ## accounting
        losses_valid /= nb_valid_batches
        for i, _ in enumerate(names):
            tsb_valid[i].add_scalar(f'loss/regression', losses_valid[i,0], idx_epoch)
            tsb_valid[i].add_scalar(f'loss/gradient', losses_valid[i,1], idx_epoch)
            tsb_valid[i].add_scalar(f'loss/gradient_dir', losses_valid[i,2], idx_epoch)
            tsb_valid[i].add_scalar(f'loss/eikonal', losses_valid[i,3], idx_epoch)
            tsb_valid[i].add_scalar(f'loss/total', losses_valid[i,:].sum(axis=-1), idx_epoch)

        ## print
        toc = time.time()
        for i, n in enumerate(names):
            print(f'average train error {n}:' + ''.join([f' {losses_train[i][j]:>.4f} /' for j in range(len(sdf_losses_weights))])[:-2])
            print(f'average valid error {n}:' + ''.join([f' {losses_valid[i][j]:>.4f} /' for j in range(len(sdf_losses_weights))])[:-2])
        print(f'epoch time: {toc-tic}')

        ## step rate scheduler
        if idx_epoch < lr_nb_steps:
            for sch in schedulers: sch.step()

        ## save
        for name, nn in zip(names, nns):
            nn.to('cpu')
            torch.jit.save(nn, f'{nn_file_now}/{name}/weights.pt')
            torch.jit.save(nn, f'{nn_file_now}/{name}/epochs/e{idx_epoch}.pt')
            nn.to(device)

    print('done!')
