import numpy as np
import torch
from .visualization import Imgs2Points


class PosSampler:
    """Parallelized 3D position sampler checker in Cartesian coordinates.
    Sampling is performed using warp.
    Returns normalized coordinates.
    The .normalize() method can be called to normalize as [x/dmax; y/dmax.atan(hfov); z/dmax.atan(vfov)].
    """
    def __init__(self, dmax, hfov, vfov, margin=20, is_spherical=False, device='cpu'):
        self.dmax = dmax
        self.hfov = hfov
        self.vfov = vfov
        self.margin = margin
        self.device = device

        self.img_to_points = Imgs2Points(False, is_spherical, dmax, hfov, vfov, remove_d0=False, remove_dmax=False, downsamp=5, device=device)

        self.atanh = np.tan(self.hfov)
        self.atanv = np.tan(self.vfov)

        ## sizes with margin
        hfov_effective = min(np.pi, self.hfov * (100 + self.margin) / 100)
        vfov_effective = min(np.pi / 2, self.vfov * (100 + self.margin) / 100)
        dinf = 0 # - (self.dmax) * (self.margin / 2) / 100
        dsup = (self.dmax) * (100 + self.margin / 2) / 100
        drange = dsup - dinf
        self.sizes_margin = (dinf, dsup, drange, hfov_effective, vfov_effective)

        ## sizes without margin
        hfov_effective = min(np.pi, self.hfov)
        vfov_effective = min(np.pi/2, self.vfov)
        dinf = 0
        dsup = self.dmax
        drange = dsup - dinf
        self.sizes_nomargin = (dinf, dsup, drange, hfov_effective, vfov_effective)


    ## normalization function
    def normalize(self, points):
        points_normalized = torch.zeros_like(points)
        points_normalized[:,0] = points[:,0] / self.dmax
        points_normalized[:,1] = points[:,1] / (self.dmax * self.atanh)
        points_normalized[:,2] = points[:,2] / (self.dmax * self.atanv)
        return points_normalized


    ## sampling in a box
    def sample_pos_in_box(self, nb_points, add_margin=False):
        """Samples nb_points points (x, y, z) into the a rectangular box containing the whole frustrum.
        Return both the points tensor and the normalized points tensor (wrt dmax, hfov and vfov).
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the box.
        """
        dinf, dsup, drange, _, _ = self.sizes_margin if add_margin else self.sizes_nomargin

        points = torch.zeros(nb_points, 3, dtype=torch.float32, device=self.device)
        points[:,0] = torch.rand(nb_points) * drange + dinf
        points[:,1] = torch.rand(nb_points) * 2 * dsup - dsup
        points[:,2] = torch.rand(nb_points) * 2 * dsup - dsup

        return points


    ## sampling in a ball
    def sample_pos_in_ball(self, nb_points, ball_size, add_margin=False):
        """Samples nb_points points (x, y, z) into the a ball centered on origin.
        Return both the points tensor and the normalized points tensor (wrt dmax, hfov and vfov).
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the ball.
        """
        if add_margin:
            ball_size = ball_size*(100+self.margin)/100

        r = torch.rand(nb_points)**(1/3) * ball_size
        azimuth = torch.rand(nb_points) * 2 * np.pi
        inclination = torch.acos(torch.rand(nb_points) * 2 - 1)

        points = torch.zeros(nb_points, 3, dtype=torch.float32, device=self.device)
        points[:,0] = r * torch.sin(inclination) * torch.cos(azimuth)
        points[:,1] = r * torch.sin(inclination) * torch.sin(azimuth)
        points[:,2] = r * torch.cos(inclination)

        return points


    ## sampling in a frustrum
    def sample_pos_in_frustrum(self, nb_points, add_margin=False):
        """Samples nb_points points (x, y, z) that fall into the sensor frustrum, assuming it is a sphere sector.
        Return both the points tensor and the normalized points tensor (wrt dmax, hfov and vfov).
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the ball.
        """
        dinf, dsup, drange, hfov, vfov = self.sizes_margin if add_margin else self.sizes_nomargin

        r = torch.rand(nb_points)**(1/3) * drange + dinf
        azimuth = (torch.rand(nb_points) * 2 - 1) * hfov
        ## inclination = pi/2 - elevation
        inclination = torch.rand(nb_points) * 2 * vfov + (np.pi / 2 - vfov)

        points = torch.zeros(nb_points, 3, dtype=torch.float32, device=self.device)
        points[:,0] = r * torch.sin(inclination) * torch.cos(azimuth)
        points[:,1] = r * torch.sin(inclination) * torch.sin(azimuth)
        points[:,2] = r * torch.cos(inclination)

        return points


    def sample_pos_in_frustrum_margin(self, nb_points):
        """Samples nb_points points (x, y, z) that fall into the sensor frustrum, assuming it is a sphere sector.
        Return both the points tensor and the normalized points tensor (wrt dmax, hfov and vfov).
        """
        dinf, dsup_margin, drange_margin, hfov_margin, vfov_margin = self.sizes_margin
        dinf, dsup_nomargin, drange_nomargin, hfov_nomargin, vfov_nomargin = self.sizes_nomargin

        r = torch.zeros(nb_points, dtype=torch.float32, device=self.device)
        azimuth = torch.zeros(nb_points, dtype=torch.float32, device=self.device)
        inclination = torch.zeros(nb_points, dtype=torch.float32, device=self.device)

        n = nb_points // 5

        ## sample in +hfov margin
        r[:n] = torch.rand(n)**(1/3) * drange_margin + dinf
        azimuth[:n] = torch.rand(n) * (hfov_margin - hfov_nomargin) + hfov_nomargin
        inclination[:n] = torch.rand(n) * 2 * vfov_margin + (np.pi / 2 - vfov_margin)

        # ## sample in -hfov margin
        r[n:2*n] = torch.rand(n)**(1/3) * drange_margin + dinf
        azimuth[n:2*n] = - (torch.rand(n) * (hfov_margin - hfov_nomargin) + hfov_nomargin)
        inclination[n:2*n] = torch.rand(n) * 2 * vfov_margin + (np.pi / 2 - vfov_margin)

        # ## sample in +vfov margin
        r[n*2:3*n] = torch.rand(n)**(1/3) * drange_margin + dinf
        azimuth[n*2:3*n] = (torch.rand(n) * 2 - 1) * hfov_margin
        inclination[n*2:3*n] = torch.rand(n) * (vfov_margin - vfov_nomargin) + (np.pi / 2 - vfov_nomargin)

        # ## sample in -vfov margin
        r[3*n:4*n] = torch.rand(n)**(1/3) * drange_margin + dinf
        azimuth[3*n:4*n] = (torch.rand(n) * 2 - 1) * hfov_margin
        inclination[3*n:4*n] = (torch.rand(n) * (vfov_nomargin - vfov_margin) + (np.pi / 2 + vfov_margin))

        # ## sample in +dsup margin
        r[4*n:] = torch.rand(n)**(1/3) * (dsup_margin - dsup_nomargin) + dsup_nomargin
        azimuth[4*n:] = (torch.rand(n) * 2 - 1) * hfov_nomargin
        inclination[4*n:] = torch.rand(n) * 2 * vfov_nomargin + (np.pi / 2 - vfov_nomargin)

        points = torch.zeros(nb_points, 3, dtype=torch.float32, device=self.device)
        points[:,0] = r * torch.sin(inclination) * torch.cos(azimuth)
        points[:,1] = r * torch.sin(inclination) * torch.sin(azimuth)
        points[:,2] = r * torch.cos(inclination)

        return points


    ## sample close to obstacles
    def sample_pos_around_obs(self, imgs, points_per_img, mode='closest', std=0.2):
        """Samples points_per_img points (x, y, z) around the surface of visible obstacles in each images.
        A subsample of the points in the image pointcloud are perturbed with a Gaussian noise N(0, std).
        mode        -- selection mode ('closest' or 'random')
        std         -- std for point noising [meters]
        """
        points = self.img_to_points(imgs)

        if mode == 'random':
            if points.ndim == 2:
                points = points[torch.randint(points.shape[-2], (points_per_img,))]
            else:  # handle batch dimension
                points = points.transpose(0,1)[torch.randint(points.shape[-2], (points_per_img,))].transpose(0,1)
        elif mode == 'closest':
            if points.shape[-2] < points_per_img:
                raise AssertionError('too few points for sampling around obs, reduce downsamp value')
            norms = torch.norm(points, dim=-1)
            sorted_indices = norms.argsort(dim=-1)[..., :points_per_img]
            points = torch.gather(points, dim=-2, index=sorted_indices.unsqueeze(-1).expand(-1, -1, 3))

        return points + torch.randn_like(points) * std


    ## grids
    def grid_frustrum_slice(self, nb_points, elevation_deg, add_margin=False, h360=False):
        """Generates a grid of ~nb_points of uniformly spaced points over a sensor frustrum "slice", i.e. for a given vertical bearing.
        The grid has floor(nb_points**(1/2)) points in each dimension, so if nb_points is not a perfect square, it does not correspond to the output shape.
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the sensor frustrum.
        The h360 param allows to sample with an infinite horizontal fov.
        """
        dinf, dsup, drange, hfov, vfov = self.sizes_margin if add_margin else self.sizes_nomargin
        grid_size = round(nb_points**(1/2))

        if h360: hfov = np.pi

        r = np.repeat(np.linspace(dinf, dsup, grid_size), grid_size)
        azimuth = np.tile(np.linspace(-hfov, hfov, grid_size, dtype=np.float32), grid_size)
        inclination = np.pi / 2 - elevation_deg * np.pi / 180

        points = np.zeros((grid_size**2, 3), dtype=np.float32)
        points[:,0] = r * np.sin(inclination) * np.cos(azimuth)
        points[:,1] = r * np.sin(inclination) * np.sin(azimuth)
        points[:,2] = r * np.cos(inclination)

        points = torch.from_numpy(points).to(self.device)
        return points


    def grid_frustrum(self, nb_points, add_margin=False):
        """Generates a grid of ~nb_points of uniformly spaced points over the sensor frustrum.
        The grid has floor(nb_points**(1/3)) points in each dimension, so if nb_points is not a perfect cube, it does not correspond to the output shape.
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the sensor frustrum.
        """
        dinf, dsup, drange, hfov, vfov = self.sizes_margin if add_margin else self.sizes_nomargin
        grid_size = round(nb_points**(1/3))

        r = np.repeat(np.linspace(dinf, dsup, grid_size), grid_size**2)
        azimuth = np.repeat(np.tile(np.linspace(-hfov, hfov, grid_size), grid_size), grid_size)
        ## inclination = pi/2 - elevation, so abs(elevation) < vfov <=> abs(inclination) < pi/2 + vfov <=> cos(inclination) < cos(pi/2 + vfov) = sin(vfov)
        inclination = np.tile(np.arccos(np.linspace(-np.sin(vfov), np.sin(vfov), grid_size)), grid_size**2)

        points = np.zeros((grid_size**3, 3), dtype=np.float32)
        points[:,0] = r * np.sin(inclination) * np.cos(azimuth)
        points[:,1] = r * np.sin(inclination) * np.sin(azimuth)
        points[:,2] = r * np.cos(inclination)

        points = torch.from_numpy(points).to(self.device)
        return points


    def grid_sphere(self, nb_points, add_margin=False):
        """Generates a grid of ~nb_points of uniformly spaced (along each dimension) points over a sphere containing the fov volume.
        The grid has floor(nb_points**(1/3)) points in each dimension, so if nb_points is not a perfect cube, it does not correspond to the output shape.
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the sphere.
        """
        dinf, dsup, drange, _, _ = self.sizes_margin if add_margin else self.sizes_nomargin
        grid_size = int(nb_points**(1/3))

        r = np.repeat(np.linspace(dinf, dsup, grid_size), grid_size**2)
        azimuth = np.repeat(np.tile(np.linspace(-np.pi, np.pi, grid_size), grid_size), grid_size)
        inclination = np.tile(np.arccos(np.linspace(-1, 1, grid_size)), grid_size**2)

        points = np.zeros((grid_size**3, 3), dtype=np.float32)
        points[:,0] = r * np.sin(inclination) * np.cos(azimuth)
        points[:,1] = r * np.sin(inclination) * np.sin(azimuth)
        points[:,2] = r * np.cos(inclination)

        points = torch.from_numpy(points).to(self.device)
        return points


    def grid_sphere_fixed_step(self, step, in_frustrum=False, frustrum_is_spherical=False, add_margin=False):
        """Generates a grid of evenly spaced points over a box containing the fov volume.
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the box.
        """
        dinf, dsup, drange, hfov, vfov = self.sizes_margin if add_margin else self.sizes_nomargin

        dsup = np.round(dsup/step)*step
        dinf = -dsup
        x = np.arange(dinf, dsup*1.001, step)
        y = np.arange(-dsup, dsup*1.001, step)
        z = np.arange(-dsup, dsup*1.001, step)

        points = np.zeros((x.shape[0] * y.shape[0] * z.shape[0], 3), dtype='float32')
        points[:,0] = np.repeat(x, y.shape[0] * z.shape[0])
        points[:,1] = np.tile(y.repeat(x.shape[0]), z.shape[0])
        points[:,2] = np.tile(z, x.shape[0] * y.shape[0])
        points = np.unique(points, axis=0)

        if in_frustrum:
            points = points[np.linalg.norm(points, axis=1) <= dsup*1.001, :]
            points = points[np.abs(np.arctan2(points[:,1], points[:,0])) <= hfov*1.001, :]
            if frustrum_is_spherical:
                points = points[np.abs(np.arctan2(points[:,2], np.linalg.norm(points[:,:2], axis=1))) <= vfov*1.001, :]
            else:
                points = points[np.abs(np.arctan2(points[:,2], points[:,0])) <= vfov*1.001, :]

        points = torch.from_numpy(points).to(torch.float32).to(self.device)
        return points
