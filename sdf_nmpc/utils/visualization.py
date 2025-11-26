import numpy as np
import torch


class Imgs2Points(torch.nn.Module):
    """Transform depth or range images to pointclouds.
    Assumes dmax-normalized [H, W], [C, H, W] or [B, C, H, W] tensor.
    is_depth    -- pixel value is depth instead of range
    is_spherical-- pixel coordinates are spherical instead of Cartesian
    hfov        -- horizontal fov
    vfov        -- vertical fov
    remove_dmax -- remove points at max distance from point cloud
    downsamp    -- downsampling ratio (apply spatial minpooling over input)
    device      -- a torch.device
    """
    def __init__(self, is_depth, is_spherical, dmax, hfov, vfov, downsamp=1, remove_d0=False, remove_dmax=False, device='cpu'):
        super(Imgs2Points, self).__init__()
        self.is_depth = is_depth
        self.is_spherical = is_spherical
        self.dmax = dmax
        self.hfov = hfov
        self.vfov = vfov
        self.remove_d0 = remove_d0
        self.remove_dmax = remove_dmax
        self.downsamp = downsamp
        self.device = device


    def forward(self, imgs):
        ## create chan dim if missing
        if imgs.ndim == 2: imgs = imgs.unsqueeze(0)

        ## minpool
        if self.downsamp != 1:
            imgs = -torch.nn.functional.max_pool2d(-imgs, kernel_size=self.downsamp)

        ## creates a y,z meshgrid of image plane
        height = imgs.shape[-2]
        width = imgs.shape[-1]
        u = torch.arange(0, width, dtype=torch.float32, device=self.device)
        v = torch.arange(0, height, dtype=torch.float32, device=self.device)
        u, v = torch.meshgrid(u, v, indexing='xy')

        ## get 3D coordinate of pixel
        if self.is_spherical:
            p = self._pixel_grid_spherical(u, v)
        else:
            p = self._pixel_grid_cartesian(u, v)
            if not self.is_depth:
                p = p / torch.linalg.vector_norm(p, dim=0)

        ## handle batch dimension
        if imgs.ndim == 3:
            p = (p * imgs * self.dmax).reshape(3,-1).T
        else:
            nb_batch = imgs.shape[0]
            p = p.unsqueeze(0).repeat((nb_batch,1,1,1))
            p = (p * imgs * self.dmax).reshape(nb_batch,3,-1).transpose(1,2)

        ## remove points at d == 0
        if self.remove_d0:
            if self.is_depth:
                p = p[p[...,0] > 0.01]
            else:
                p = p[torch.linalg.vector_norm(p, dim=-1) > 0.01]

        ## remove points at d == dmax
        if self.remove_dmax:
            if self.is_depth:
                p = p[p[...,0] < self.dmax*0.99]
            else:
                p = p[torch.linalg.vector_norm(p, dim=-1) < self.dmax*0.99]

        if imgs.ndim == 3:
            return p.view((-1, 3))
        else:
            return p.view((nb_batch, -1, 3))


    def _pixel_grid_cartesian(self, u, v):
        """Create array of Cartesian pixel positions.
        Linear interpolation in the plane z = 1, between y(u=0) = tan(hfov) et y(u=width) = -tan(hfov).
        Similarly on the z axis, using the corresponding fov value.
        """
        hwidth = u.shape[1] / 2
        hheight = u.shape[0] / 2

        x = torch.ones_like(u)
        y = np.tan(self.hfov) * (1 - u / hwidth)
        z = np.tan(self.vfov) * (1 - v / hheight)
        return torch.stack([x, y, z], dim=0)


    def _pixel_grid_spherical(self, u, v):
        """Create array of spherical pixel positions.
        Linear interpolation on the sphere range = 1, between azimuth(u=0) = hfov et azimuth(u=width) = -hfov.
        Similarly on the z axis, using the corresponding fov value.
        """
        hwidth = u.shape[1] / 2
        hheight = u.shape[0] / 2

        ## get azimuth and elevation angles from u, v
        azimuth = self.hfov * (1 - u / hwidth)
        elevation = self.vfov * (1 - v / hheight)

        x = torch.cos(elevation) * torch.cos(azimuth)
        y = torch.cos(elevation) * torch.sin(azimuth)
        z = torch.sin(elevation)
        return torch.stack([x, y, z], dim=0)
