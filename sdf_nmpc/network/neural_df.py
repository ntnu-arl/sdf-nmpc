import numpy as np
import torch
from ..utils.embeddings import PositionEmbedding
from ..utils.activation import Sine


class NeuralDF(torch.nn.Module):
    """Neural DF.
    Either signed or unsigned (SDF or UDF), depending on `signed` argument in init.
    The NN computes the (truncated) DF at a given 3D position, given a depth image compressed into a latent space.
    At inference time, the NN is (totally or partially) written in ml-casadi, in order to be integrated into the NMPC, with the latent vector passed as parameters.
    """
    def __init__(
        self,
        nb_states=3,
        size_latent=128,
        signed=True,
        max_df=1,
        res='full',
        w0=1.0,
        embed='pos',
        act='sin',
        layer_sizes=[256, 256, 256, 256],
        dropout_rate=None,
        nb_freqs=5,
    ):
        super(NeuralDF, self).__init__()
        self.nb_states = nb_states
        self.size_latent = size_latent
        self.signed = signed
        self.res = res
        self.w0 = w0
        self.max_df = max_df
        self.activation = act
        self.embeddings = embed
        self.dropout_rate = dropout_rate
        self.nb_freqs = nb_freqs

        ## activation function
        if act == 'sin':
            self.act = Sine(self.w0)
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'softplus':
            self.act = torch.nn.Softplus()
        else:
            raise AttributeError('Unknown activation function')

        ## embeddings
        avail_proj = {'pos':'none', 'cube':'cube', 'oct':'octohedron', 'dod':'dodecahedron', 'ico':'icosahedron'}
        if embed == 'none':
            self.embed = torch.nn.Identity()
            self.nb_embeddings = 3
        elif embed in avail_proj:
            self.embed = PositionEmbedding(self.nb_freqs, proj=avail_proj[embed])
            self.nb_embeddings = self.embed.nb_embeddings
        else:
            raise AttributeError('Unknown embedding function')

        ## layers
        self.layers = torch.nn.ModuleDict({
            'embeddings': torch.nn.Sequential(
                self.embed,
            ),
            'main1': torch.nn.Sequential(
                torch.nn.Linear(self.nb_embeddings + self.size_latent, layer_sizes[0]),
                self.act,
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(layer_sizes[0], layer_sizes[1]),
                self.act,
                torch.nn.Dropout(dropout_rate),
            ),
            'main2': torch.nn.Sequential(
                torch.nn.Linear(
                    layer_sizes[1]
                        + (self.nb_embeddings + self.size_latent) * (self.res == 'full')
                        + (self.nb_embeddings) * (self.res == 'state')
                        + (self.size_latent) * (self.res == 'latent'),
                    layer_sizes[2]),
                self.act,
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(layer_sizes[2], layer_sizes[3]),
                self.act,
                torch.nn.Dropout(dropout_rate),
            ),
            'df': torch.nn.Sequential(
                torch.nn.Linear(layer_sizes[3], 1),
            ),
        })

    def forward(self, x):
        state = x[:, :3]
        latent = x[:, 3:]
        embeddings = self.layers['embeddings'](state)
        x = torch.concatenate((embeddings, latent), 1)
        x = self.layers['main1'](x)
        if self.res in ['full', 'state']:
            x = torch.concatenate((x, embeddings), 1)
        if self.res in ['full', 'latent']:
            x = torch.concatenate((x, latent), 1)
        x = self.layers['main2'](x)
        df = self.layers['df'](x)
        return df
