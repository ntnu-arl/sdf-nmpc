import os
import time
import datetime
import argparse
import numpy as np
import torch
from torchinfo import summary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # shuts down warning message when loading tensorboard
from torch.utils.tensorboard import SummaryWriter
from sdf_nmpc.network.vae import Vae
from sdf_nmpc.utils.losses import loss_MSE_valid_pixels, loss_MSE_valid_pixels_bias_pos_dist, loss_KLD
from sdf_nmpc.utils.data import train_dataset_from_h5
from sdf_nmpc.utils.layer_init import init_conv_layers

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _paths import path_to_data, path_to_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='device', nargs='?', default='cuda', help='Device to use.')
    device = parser.parse_args().device
    print('using', device)

    ## parameters64
    # dataset = '_virtual_depth_full.hdf5'
    # vae_file = 'vae'
    dataset = '_virtual_lidar_full.hdf5'
    vae_file = 'vae_lidar'
    train_valid_ratio = 0.9
    size_latent = 64
    nb_epochs = 200
    lr_start = 2e-4
    lr_min = 5e-5
    lr_nb_steps = 50
    weight_decay = 1e-5
    batch_size_train = 50
    batch_size_valid = 300
    dmax = 5
    beta_kld = 1
    bias = True
    bias_dist_ratio = 0.1
    bias_dist_degree = 3
    bias_pos_ratio = 0.1
    restart_from_epoch = 0

    ## data
    (train_dataloader, valid_dataloader), metadata = train_dataset_from_h5(
        path_to_data, dataset, dmax, batch_size_train=batch_size_train,
        batch_size_valid=batch_size_valid, train_valid_ratio=train_valid_ratio,
        vae=True, col_map=True, device=device, seed=42,
    )

    shape_imgs = metadata['shape_imgs']

    ## nn
    if restart_from_epoch:
        now = '2025-04-22T19:04:47'
    else:
        now = datetime.datetime.now().isoformat()[:-7]
    nn_file_now = os.path.join(path_to_weights, vae_file, now)

    if restart_from_epoch:
        vae = torch.jit.load(f'{nn_file_now}/epochs/e{restart_from_epoch-1}.pt')
    else:
        os.makedirs(f'{nn_file_now}/epochs')
        vae = Vae(size_latent=size_latent, shape_imgs=shape_imgs[1:], dropout_rate=0.1, batchnorm=True)
        init_conv_layers(vae.encoder.layers)
        init_conv_layers(vae.decoder.layers)
        vae = torch.jit.script(vae)

    vae.to(device)

    ## nn summary
    summary(vae, input_size=[1]+shape_imgs, depth=4, device=device)

    ## tensorboard writers
    tsb_train = SummaryWriter(log_dir=os.path.join(nn_file_now, 'train'))
    tsb_valid = SummaryWriter(log_dir=os.path.join(nn_file_now, 'valid'))

    ## training printing parameters
    nb_train_batches = len(train_dataloader)
    nb_batches_print_train = int(nb_train_batches/5)
    nb_valid_batches = len(valid_dataloader)
    nb_batches_print_valid = int(nb_valid_batches/2)

    ## optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_nb_steps, eta_min=lr_min)
    ## step scheduler if restarting from epoch
    for _ in range(restart_from_epoch):
        scheduler.step()

    ## training loop
    for idx_epoch in range(restart_from_epoch, nb_epochs):
        tic = time.time()
        print(f'-------------------------------\nepoch {idx_epoch} -- lr: {scheduler.get_last_lr()[0]:.2e}')
        losses_train = np.zeros(2)  # aggregates losses across batches
        losses_valid = np.zeros(2)

        ## train
        vae.train()
        for idx_batch, (imgs_in, imgs_out) in enumerate(train_dataloader):
            optimizer.zero_grad()

            ## forward & loss
            mean, logvar = vae.encoder.mean_logvar(imgs_in)
            latent = vae.encoder.sample(mean, logvar)
            nn_output = vae.decoder(latent)
            if bias:
                loss_regr = loss_MSE_valid_pixels_bias_pos_dist(imgs_out, nn_output, bias_pos_ratio, bias_dist_ratio, bias_dist_degree)
            else:
                loss_regr = loss_MSE_valid_pixels(imgs_out, nn_output)
            loss_kld = loss_KLD(mean, logvar, beta_kld, size_latent, shape_imgs[1:])

            losses_train[0] += loss_regr.item()
            losses_train[1] += loss_kld.item()

            ## backpropagation
            loss = loss_regr + loss_kld
            loss.backward()
            optimizer.step()

            ## print
            print(f'train batch: {idx_batch+1}/{nb_train_batches} - loss (reg/kld): {loss_regr.item():>4f} / {loss_kld.item():>4f} - elapsed time: {(time.time()-tic):.3f}', end='\r')
            if not ((idx_batch+1) % nb_batches_print_train): print()

        ## accounting
        losses_train /= nb_train_batches
        tsb_train.add_scalar(f'loss/regression', losses_train[0], idx_epoch)
        tsb_train.add_scalar(f'loss/kld', losses_train[1], idx_epoch)
        tsb_train.add_scalar(f'loss/total', losses_train.sum(), idx_epoch)

        ## valid
        vae.eval()
        with torch.no_grad():
            for idx_batch, (imgs_in, imgs_out) in enumerate(valid_dataloader):
                ## forward & loss
                mean, logvar = vae.encoder.mean_logvar(imgs_in)
                nn_output = vae.decoder(mean)
                if bias:
                    loss_regr = loss_MSE_valid_pixels_bias_pos_dist(imgs_out, nn_output, bias_pos_ratio, bias_dist_ratio, bias_dist_degree)
                else:
                    loss_regr = loss_MSE_valid_pixels(imgs_out, nn_output)
                loss_kld = loss_KLD(mean, logvar, beta_kld, size_latent, shape_imgs[1:])

                losses_valid[0] += loss_regr.item()
                losses_valid[1] += loss_kld.item()

                print(f'valid batch: {idx_batch+1}/{nb_valid_batches} - loss (reg/kld): {loss_regr.item():>4f} / {loss_kld.item():>4f} - elapsed time: {(time.time()-tic):.3f}', end='\r')
                if not ((idx_batch+1) % nb_batches_print_valid): print()

        ## accounting
        losses_valid /= nb_valid_batches
        tsb_valid.add_scalar(f'loss/regression', losses_valid[0], idx_epoch)
        tsb_valid.add_scalar(f'loss/kld', losses_valid[1], idx_epoch)
        tsb_valid.add_scalar(f'loss/total', losses_valid.sum(), idx_epoch)

        ## print
        toc = time.time()
        print(f'average train error (reg,kld): {losses_train[0].mean():>4f} , {losses_train[1].mean():>4f}')
        print(f'average valid error (reg,kld): {losses_valid[0].mean():>4f} , {losses_valid[1].mean():>4f}')
        print(f'epoch time: {toc-tic}')

        ## step rate scheduler
        scheduler.step()

        ## save
        vae.to('cpu')
        torch.jit.save(vae, f'{nn_file_now}/weights.pt')
        torch.jit.save(vae, f'{nn_file_now}/epochs/e{idx_epoch}.pt')
        vae.to(device)

    print('done!')
