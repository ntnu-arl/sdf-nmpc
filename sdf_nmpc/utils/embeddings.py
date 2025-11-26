import numpy as np
import torch
import casadi as cs


class PositionEmbedding(torch.nn.Module):
    """Implementation of the Off-Axis Positional Embedding described in [Barron et al., 2022] -- see https://arxiv.org/abs/2111.12077
    The implementation is mostly taken from iSDF paper [Ortiz et al., 2022] -- see https://joeaortiz.github.io/iSDF/
    Code reference: https://github.com/facebookresearch/iSDF/blob/main/isdf/modules/embedding.py
    The output feature vector is [x, sin(2^iAx), cos(2^iAx)], i in {0...nb_freqs}
    """
    def __init__(self, nb_freqs=10, proj='none'):
        """PositionEmbedding class constructor.
        The off-axis projection mode is either 'none', 'icosahedron' or 'dodecahedron'.
            - 'none':  classical in-axis PE described in [Barron et al., 2021] -- see https://arxiv.org/abs/2103.13415
            - 'icosahedron': icosahedron off-axis projection described in [Barron et al., 2022], used in [Ortiz et al., 2022]
            - 'dodecahedron': dodecahedron off-axis projection, for the sake of reducing the number of output features
        """
        super(PositionEmbedding, self).__init__()
        self.nb_freqs = nb_freqs
        freq_bands = 2 ** torch.linspace(0, nb_freqs - 1, nb_freqs, dtype=torch.float32)

        ## direction of projections
        if proj == 'none':
            dirs = torch.eye(3, dtype=torch.float32)
        elif proj == 'cube':
            ## the projection directions are the centroids of the 6 faces of a regular cube
            phi = (1 + np.sqrt(5)) / 2
            dirs = torch.tensor([
                -1, 0, 0,
                +1, 0, 0,
                0, -1, 0,
                0, +1, 0,
                0, 0, -1,
                0, 0, +1,
            ], dtype=torch.float32).reshape(-1, 3).T
        elif proj == 'octohedron':
            ## the projection directions are the centroids of the 8 faces of a regular octohedron
            ## ie, the vertices of a regular cube
            phi = (1 + np.sqrt(5)) / 2
            dirs = torch.tensor([
                -1, -1, -1,
                -1, -1, +1,
                -1, +1, -1,
                -1, +1, +1,
                +1, -1, -1,
                +1, -1, +1,
                +1, +1, -1,
                +1, +1, +1,
            ], dtype=torch.float32).reshape(-1, 3).T
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=0)
        elif proj == 'dodecahedron':
            ## the projection directions are the centroids of the 12 faces of a regular dodecahedron
            ## ie, the vertices of a regular icosahedron -- https://en.wikipedia.org/wiki/Regular_icosahedron#Construction
            phi = (1 + np.sqrt(5)) / 2
            dirs = torch.tensor([
                0, -1, -phi,
                0, +1, -phi,
                0, -1, +phi,
                0, +1, +phi,
                -1, 0, -phi,
                +1, 0, -phi,
                -1, 0, +phi,
                +1, 0, +phi,
                -1, -phi, 0,
                +1, -phi, 0,
                -1, +phi, 0,
                +1, +phi, 0,
            ], dtype=torch.float32).reshape(-1, 3).T
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=0)
        elif proj == 'icosahedron':
            ## the projection directions are the centroids of the 20 faces of a regular icosahedron
            ## ie, the vertices of a regular dodecahedron -- https://en.wikipedia.org/wiki/Regular_dodecahedron#Relation_to_the_golden_ratio
            phi = (1 + np.sqrt(5)) / 2
            h = 1 / phi
            dirs = torch.tensor([
                +1, +1, +1,
                +1, +1, -1,
                +1, -1, +1,
                +1, -1, -1,
                -1, +1, +1,
                -1, +1, -1,
                -1, -1, +1,
                -1, -1, -1,
                0, +phi, +h,
                0, +phi, -h,
                0, -phi, +h,
                0, -phi, -h,
                +h, 0, +phi,
                +h, 0, -phi,
                -h, 0, +phi,
                -h, 0, -phi,
                +phi, +h, 0,
                +phi, -h, 0,
                -phi, +h, 0,
                -phi, -h, 0,
            ], dtype=torch.float32).reshape(-1, 3).T
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=0)
        else:
            raise AttributeError('Unknown off-axis projection mode')

        self.register_buffer('freq_bands', freq_bands)
        self.register_buffer('dirs', dirs)
        self.nb_embeddings = self.nb_freqs * dirs.shape[-1] * 2 + 3

    def forward(self, x):
        proj = torch.matmul(x, self.dirs)
        xb = torch.reshape(proj[..., None] * self.freq_bands, list(proj.shape[:-1]) + [-1])
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([x] + [embedding], dim=-1)
        return embedding
