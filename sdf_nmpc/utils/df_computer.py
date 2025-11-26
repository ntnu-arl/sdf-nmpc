import numpy as np
import warp as wp
import torch
from .collision_checker import ColChecker


class DfComputer:
    """Parallelized unsigned distance function computer from depth or range images."""
    def __init__(self, signed, dmax, hfov, vfov, max_df, is_depth=False, is_spherical=False, batch_size=5000, device=None):
        self.signed = signed
        self.is_depth = is_depth
        self.is_spherical = is_spherical
        self.hfov = hfov
        self.vfov = vfov
        self.dmax = dmax
        self.min_df = -0.3  # assuming min < max
        self.max_df = 1
        wp.init()
        self.device = device if device is not None else str(wp.get_device())

        if self.signed:
            self.colcheck = ColChecker(dmax, hfov, vfov, 0, is_depth, is_spherical, 'extrapolate', device=self.device)

            ## get grid and associated distances
            # grid_params = [(0, 0.1, 0.01), (0.1, 0.3, 0.025), (0.3, max_df, 0.05)]
            # grid_params = [(0, 0.1, 0.01), (0.1, 0.3, 0.02), (0.3, 0.5, 0.04), (0.5, 0.7, 0.05), (0.7, 1, 0.1)]
            grid_params = [(0, 0.1, 0.01), (0.1, 0.2, 0.02), (0.2, 0.3, 0.03), (0.3, 0.5, 0.05), (0.5, 1, 0.1)]
            self.batch_size = batch_size

            self.distances, self.grid = self.generate_dist_grid(grid_params)


    def generate_dist_grid(self, grid_params):
        """ Compute a voxel grid for computing SDF wrt its center
        The voxel size is typically increasing as the distance to the origin increases for efficiency.
        gris_params is a list of tuples (dmin, dmax, size), such that the grid resolution is, eg:
            0    - 0.05m -> 1cm
            0.05 - 0.1m  -> 2cm
            ...
        """
        distances = torch.tensor([], dtype=torch.float32, device=self.device)
        grid = torch.tensor([], dtype=torch.float32, device=self.device)
        for (dmin, dmax, step) in grid_params:
            nb_points = int(2.*dmax/step) + 1
            coords = torch.linspace(-dmax, dmax, nb_points, device=self.device)
            local_grid = torch.stack([
                coords.repeat(nb_points).repeat(nb_points),
                coords.repeat_interleave(nb_points).repeat(nb_points),
                coords.repeat_interleave(nb_points).repeat_interleave(nb_points),
            ], dim=1)
            local_dist = torch.linalg.norm(local_grid, dim=1)

            idx = (local_dist > dmin) * (local_dist <= dmax)
            grid = torch.cat([grid, local_grid[idx]], dim=0)
            distances = torch.cat([distances, local_dist[idx]], dim=0)

        return distances, grid


    def get_df(self, imgs, points, p_to_i=None):
        """Compute the signed or unsigned ditance field for points according to imgs.
        Size of image array: [B, H, W] or [H, W].
        Images are assumed normalized wrt dmax.
        Points are unormalized Cartesian coordinates.
        p_to_i is an array mapping each point to the index of the corresponding image. If set to None, default ordering is assumed.
        """
        if not torch.is_tensor(imgs): imgs = torch.from_numpy(imgs).to(self.device)
        if imgs.ndim == 2: imgs = imgs.unsqueeze(0)
        if imgs.ndim != 3: raise AssertionError('input image must have size [B, H, W] or [H, W]')
        if p_to_i is None:
            p_to_i = torch.ones(points.shape[0], dtype=torch.int16, device=self.device).reshape((imgs.shape[0],-1))  # put all index for same image in rows
            p_to_i = torch.arange(imgs.shape[0], dtype=torch.int16, device=self.device)[:,None] * p_to_i
            p_to_i = p_to_i.reshape(-1)
        elif not torch.is_tensor(p_to_i):
            p_to_i = torch.from_numpy(p_to_i).to(torch.int16).to(self.device)
        if not torch.is_tensor(points):
            points = torch.from_numpy(points).to(torch.float32).to(self.device)

        if self.signed:
            return self.get_sdf(imgs, points, p_to_i)
        else:
            return self.get_udf(imgs, points, p_to_i)


    @staticmethod
    @wp.kernel
    def _kernel_pixel_wise_udf(img: wp.array3d(dtype=wp.float32), points: wp.array(dtype=wp.vec3), p_to_i: wp.array(dtype=wp.int16),
                               dmax: float, hfov: float, vfov: float, is_depth: wp.bool, is_spherical: wp.bool,
                               dist: wp.array2d(dtype=wp.float32), pix_points: wp.array2d(dtype=wp.vec3)):
        """Warp kernel function to parallelly compute the distance of a given point to each pixels in the image.
        img             -- batch of B images, array of size (B, H, W)
        points          -- array of N points [x, y, z] to be checked in the images
        p_to_i          -- point to image idx map
        dmax            -- max distance
        hfov            -- horizon half fov
        vfov            -- vertical half fov
        is_depth        -- pixel value is depth instead of range
        is_spherical    -- pixel coordinates are spherical instead of Cartesian
        dist            -- output array of size (N, HxW) of distances to pixels
        pix_points      -- output array of size (N, HxW) of 3D position of pixels wrt points
        """
        nb_pix_h = img.shape[2]
        nb_pix_v = img.shape[1]

        ## thread index
        tid = wp.tid()

        ## get corresponding point, image and pixel indices
        p_idx = tid // (nb_pix_h * nb_pix_v)
        p = points[p_idx]

        img_idx = int(p_to_i[p_idx])

        pix_idx = tid % (nb_pix_h * nb_pix_v)
        pixel_h = pix_idx % nb_pix_h
        pixel_v = pix_idx // nb_pix_h

        ## compute 3D position of pixel
        w = float(img.shape[2])
        h = float(img.shape[1])
        if is_spherical:
            x = 1.
            y = wp.tan(hfov) * (1. - 2. * float(pixel_h) / w)
            z = wp.tan(vfov) * (1. - 2. * float(pixel_v) / h)
            pix = wp.vec3(x, y, z) * img[img_idx, pixel_v, pixel_h] * dmax
        else:
            azimuth = hfov * (1. - 2. * float(pixel_h) / w)
            elevation = vfov * (1. - 2. * float(pixel_v) / h)
            x = wp.cos(elevation) * wp.cos(azimuth)
            y = wp.cos(elevation) * wp.sin(azimuth)
            z = wp.sin(elevation)
            pix = wp.vec3(x, y, z) * img[img_idx, pixel_v, pixel_h] * dmax

        pix_points[p_idx, pix_idx] = pix - p

        if x == 0:  # invalid pixel, setting dummy value
            dist[p_idx, pix_idx] = dmax
        else:
            d_p = wp.length(pix_points[p_idx, pix_idx])
            if is_depth:
                d_bg = dmax - p[0]
            else:
                d_bg = dmax - wp.length(p)

            if d_p > d_bg:  # the point is closest to the dmax virtual wall than to any point
                pix_points[p_idx, pix_idx] = wp.vec3(dmax, p[1], p[2])
                dist[p_idx, pix_idx] = d_bg
            else:
                dist[p_idx, pix_idx] = d_p


    def get_udf(self, imgs, points, p_to_i):
        ## minpool for efficiency, but disregarding 0s when computing the min
        kernel = 5  # minpool kernel, make sure it divides both dimensions
        assert (imgs.shape[1] % kernel) == 0 and (imgs.shape[2] % kernel) == 0
        min_pooled_size = np.floor(np.array([imgs.shape[1]/kernel, imgs.shape[2]/kernel])).astype(np.int32)
        min_pooled = imgs.unfold(1, kernel, kernel).unfold(2, kernel, kernel).contiguous().view([imgs.shape[0]] + list(min_pooled_size) + [kernel**2])
        idx_notfull_zeros = torch.count_nonzero(min_pooled, dim=-1) != 0
        tmp = min_pooled[idx_notfull_zeros]
        tmp[min_pooled[idx_notfull_zeros] == 0] = self.dmax
        min_pooled[idx_notfull_zeros] = tmp
        min_pooled = torch.amin(min_pooled, dim=-1)

        ## point-to-pixel distances
        min_pooled = wp.from_torch(min_pooled)
        points_wp = wp.from_torch(points, dtype=wp.vec3)
        p_to_i_wp =  wp.from_torch(p_to_i, dtype=wp.int16)
        udf_per_pixel = wp.array(shape=(points_wp.shape[0], min_pooled.shape[2] * min_pooled.shape[1]), dtype=wp.float32, device=self.device, owner=True)
        pixel_points = wp.array(shape=(points_wp.shape[0], min_pooled.shape[2] * min_pooled.shape[1]), dtype=wp.vec3, device=self.device, owner=True)
        wp.launch(
            kernel=self._kernel_pixel_wise_udf,
            dim=udf_per_pixel.shape[0] * udf_per_pixel.shape[1],
            inputs=[min_pooled, points_wp, p_to_i_wp, self.dmax, self.hfov, self.vfov, self.is_depth, self.is_spherical],
            outputs=[udf_per_pixel, pixel_points],
            device=self.device
        )
        mindist, indices = wp.to_torch(udf_per_pixel).min(dim=1)  # get minimum distance wrt among all pixels
        udf = torch.clamp(mindist, 0, self.max_df)
        grad_dirs = wp.to_torch(pixel_points)[torch.arange(points.shape[0]), indices]  # direction from point toward point associated with mindist (= direction of gradient, modulo sign)
        grad = - torch.where(udf[:,None] == self.max_df, 0, grad_dirs / (torch.linalg.vector_norm(grad_dirs, dim=-1)[:,None]))

        return udf, grad


    @staticmethod
    @torch.jit.script
    def _get_grid_points(idx: torch.Tensor, points: torch.Tensor, grid: torch.Tensor):
        return points[idx].repeat_interleave(grid.shape[0], 0) + grid.repeat(idx.shape[0], 1)


    @staticmethod
    @torch.jit.script
    def _get_mindist_occgrid(idx: torch.Tensor, occ_grid: torch.Tensor, sign: torch.Tensor, sign_bool: torch.Tensor, distances: torch.Tensor, max_df: float):
        occ_grid_switch = sign[idx,None] * occ_grid + sign_bool[idx,None] * torch.ones_like(occ_grid)  # [*(-1) + 1] in case of points in occupied space to retain the 0/1 grid format (1 for voxels to consider distances, 0 otherwise)
        grid_dists = torch.where(occ_grid_switch==0, max_df, distances * occ_grid_switch)

        return torch.min(grid_dists, dim=1)


    def get_sdf(self, imgs, points, p_to_i):
        sign_bool = self.colcheck.check_image_points(imgs, points, p_to_i)  # 0 for free space, 1 for occupied
        sign = 1 - 2 * sign_bool  # 1 for free space ("before" obstacles), -1 for "inside" obstacles

        mindist = torch.ones(points.shape[0], device=self.device) * self.max_df
        grad_dirs = torch.zeros(points.shape, device=self.device)

        ## grid-based sdf
        i = -1
        while (i := i+1) == 0 or idx[-1] != points.shape[0] - 1:
            idx = torch.arange(i * self.batch_size, min((i+1) * self.batch_size, points.shape[0]), device=self.device)

            grid_points = self._get_grid_points(idx, points, self.grid)
            occ_grid = self.colcheck.check_image_points(imgs, grid_points, p_to_i[idx].repeat_interleave(self.grid.shape[0], 0)).reshape(len(idx), -1)
            mindist[idx], indices = self._get_mindist_occgrid(idx, occ_grid, sign, sign_bool, self.distances, self.max_df)
            grad_dirs[idx,:] = self.grid[indices]

        sdf = torch.clamp(sign * mindist, self.min_df, self.max_df)
        grad_dirs = grad_dirs / torch.linalg.vector_norm(grad_dirs, dim=-1)[:,None]
        grad = -sign[:,None] * torch.where(((sdf == self.min_df) + (sdf == self.max_df))[:,None], 0, grad_dirs) # saturates gradients

        return sdf, grad
