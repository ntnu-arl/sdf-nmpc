import numpy as np
import torch
from .resnet import ResBlock, ResBlockDeconv


class Encoder(torch.nn.Module):
    """Image encoder into a latent space.
    The output shape is 2*size_latent since the VAE makes use of mean+std.
    """

    def __init__(self, nb_chan, size_latent, dropout_rate=0.1, batchnorm=True):
        super(Encoder, self).__init__()
        self.nb_chan = nb_chan
        self.size_latent = size_latent
        self.dropout_rate = float(dropout_rate)
        self.batchnorm = batchnorm

        self.layers = torch.nn.ModuleDict({
            'resnet': torch.nn.Sequential(
                torch.nn.Conv2d(self.nb_chan, 64, kernel_size=7, stride=2, padding=3),
                torch.nn.ELU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlock(64, 2, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                ResBlock(128, 2, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                ResBlock(256, 2, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                ResBlock(512, 1, use_batchnorm=self.batchnorm, dropout_rate=0.),
                torch.nn.AdaptiveAvgPool2d((2, 2)),
                torch.nn.Dropout2d(self.dropout_rate) if self.dropout_rate else torch.nn.Identity(),
                torch.nn.Flatten(),
            ),
            'mean': torch.nn.Linear(512 * 2 * 2, size_latent),
            'logvar': torch.nn.Linear(512 * 2 * 2, size_latent),
        })

    def forward(self, input):
        """Returns the mean of the latent distribution, for inference."""
        features = self.layers['resnet'](input)
        return self.layers['mean'](features)

    @torch.jit.export
    def mean_logvar(self, input):
        """Convenience function that returns the mean and logvar of the latent distribution."""
        features = self.layers['resnet'](input)
        mean = self.layers['mean'](features)
        logvar = self.layers['logvar'](features)
        return mean, logvar

    @torch.jit.export
    def sample(self, mean, logvar, num_samples:int=1):
        """Convenience function that samples the latent distribution from mean and logvar."""
        if num_samples == 1:
            return torch.randn_like(logvar) * torch.exp(0.5 * logvar) + mean
        else:
            B, N, M = mean.shape[0], self.size_latent, num_samples

            eps = torch.randn(B, M, N, device=mean.device)
            mean_expanded = mean.unsqueeze(1).expand(B, M, N)
            std_expanded = torch.exp(0.5 * logvar).unsqueeze(1).expand(B, M, N)

            return (std_expanded * eps + mean_expanded).reshape(B * M, N)


class Decoder(torch.nn.Module):
    """Image decoder from a latent space."""
    def __init__(self, nb_chan, size_latent, shape_imgs, dropout_rate=0.1, batchnorm=True):
        super(Decoder, self).__init__()
        self.nb_chan = nb_chan
        self.size_latent = size_latent
        self.shape_imgs = shape_imgs
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm

        self.layers = torch.nn.ModuleDict({
            'resnet': torch.nn.Sequential(
                torch.nn.Linear(size_latent, 512 * 8 * 15),
                torch.nn.ELU(),
                torch.nn.Unflatten(1, (512, 8, 15)),
                torch.nn.Dropout2d(self.dropout_rate) if self.dropout_rate else torch.nn.Identity(),
                ResBlockDeconv(512, 2, output_padding=1, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                ResBlockDeconv(256, 2, output_padding=1, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                ResBlockDeconv(128, 2, output_padding=1, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                ResBlockDeconv(64, 2, output_padding=1, use_batchnorm=self.batchnorm, dropout_rate=self.dropout_rate),
                torch.nn.ConvTranspose2d(32, self.nb_chan, kernel_size=5, stride=1, padding=2),
                torch.nn.Upsample(size=shape_imgs, mode='bilinear'),
                torch.nn.Sigmoid(),
            ),
        })

    def forward(self, input):
        return self.layers['resnet'](input)


class Vae(torch.nn.Module):
    """Variational AutoEncoder."""
    def __init__(self, size_latent, shape_imgs, dropout_rate=0.1, batchnorm=True):
        super(Vae, self).__init__()
        self.nb_chan = 1  # range images have 1 channel
        self.size_latent = size_latent
        self.shape_imgs = shape_imgs
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.encoder = Encoder(self.nb_chan, size_latent, dropout_rate, batchnorm)
        self.decoder = Decoder(self.nb_chan, size_latent, shape_imgs, dropout_rate, batchnorm)

    def forward(self, input):
        ## encode
        if self.training:
            mean, logvar = self.encoder.mean_logvar(input)
            latent = self.encoder.sample(mean, logvar)
        else:
            latent = self.encoder(input)

        ## decode
        return self.decoder(latent)
