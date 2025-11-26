import numpy as np
import warp as wp
import torch


class ColChecker:
    """Parallelized collision checker in depth or range images.
    outside             -- 'col', 'free', or 'extrapolate'
    """
    def __init__(self, dmax, hfov, vfov, safe_ball_size, is_depth=False, is_spherical=False, outside='free', device=None):
        assert outside in ['free', 'col', 'extrapolate']
        self.dmax = dmax
        self.hfov = hfov
        self.vfov = vfov
        self.safe_ball_size = safe_ball_size
        self.is_depth = is_depth
        self.is_spherical = is_spherical
        self.outside = 0 if outside == 'free' else 1 if outside == 'col' else 2
        wp.init()
        self.device = device if device is not None else str(wp.get_device())


    @staticmethod
    @wp.kernel
    def _kernel_colcheck(img: wp.array3d(dtype=wp.float32), points: wp.array(dtype=wp.vec3), p_to_i: wp.array(dtype=wp.int16),
                             dmax: float, hfov: float, vfov: float, outside: wp.uint8,
                             safe_ball: float, is_depth: wp.bool, is_spherical: wp.bool,
                             collisions: wp.array(dtype=wp.bool)):
        """Warp kernel function to parallelly check the collision ground truth for some point given an image.
        Collision values for points ouside of the frustrum is True.
        img             -- batch of B images, array of size (B, H, W)
        points          -- array of N points [x, y, z] to be checked in the images
        p_to_i          -- point to image idx map
        dmax            -- max distance
        hfov            -- horizon half fov
        vfov            -- vertical half fov
        outside         -- value for outside of fov -- 0: outside is free, 1: outside is collision, 2: extrapolate
        safe_ball       -- size of safe ball around the origin, within which the collision label is always 0
        is_depth        -- pixel value is depth instead of range
        is_spherical    -- pixel coordinates are spherical instead of Cartesian
        collisions      -- output array of size N collision labels
        """
        ## thread index
        tid = wp.tid()
        p = points[tid]

        if wp.length(p) > safe_ball:  # check if the point is outside the safeball, else its not in collision
            ## get value of interest in range or depth mode
            if is_depth:
                val = p[0]
            else:
                val = wp.length(p)

            if val >= dmax:  # past dmax is unsafe by default
                collisions[tid] = True

            else:
                azimuth = wp.atan2(p[1], p[0])
                if is_spherical:
                    elevation = wp.atan2(p[2], wp.sqrt(p[0]*p[0] + p[1]*p[1]))
                else:
                    elevation = wp.atan2(p[2], p[0])
                # elevation = wp.atan2(p[2], p[0])
                if outside == 2:  # if mode is extrapolate
                    azimuth = wp.clamp(azimuth, -hfov, hfov)
                    elevation = wp.clamp(elevation, -vfov, vfov)
                else:
                    if abs(azimuth) >= hfov or abs(elevation) >= vfov:  # if the point is outside of fov, set according to outside
                        if outside == 1:
                            collisions[tid] = True
                        # elif outside == 0: collisions[tid] = False  # collisions is filled with 0 by default
                        return

                ## pixel coordinates
                w = float(img.shape[2])
                h = float(img.shape[1])

                ## get corresponding pixel value in spherical or Cartesian coordinates
                if is_spherical:
                    u = wp.clamp(int(w / 2. * (1. - azimuth / hfov)), 0, img.shape[2]-1)
                    v = wp.clamp(int(h / 2. * (1. - elevation / vfov)), 0, img.shape[1]-1)
                else:
                    u = wp.clamp(int(w / 2. * (1. - wp.tan(azimuth) / wp.tan(hfov))), 0, img.shape[2]-1)
                    v = wp.clamp(int(h / 2. * (1. - wp.tan(elevation) / wp.tan(vfov))), 0, img.shape[1]-1)

                ## get image index
                img_idx = int(p_to_i[tid])

                if val >= img[img_idx, v, u] * dmax:
                    collisions[tid] = True


    def check_image_points(self, imgs, points, p_to_i=None):
        """Wrapper around the warp function to instantiate arrays and call the kernel.
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

        imgs = wp.from_torch(imgs, dtype=wp.float32)
        p_to_i =  wp.from_torch(p_to_i, dtype=wp.int16)
        points = wp.from_torch(points, dtype=wp.vec3)
        col_labels = wp.zeros((points.shape[0],), dtype=wp.bool, device=self.device)

        wp.launch(
            kernel=self._kernel_colcheck,
            dim=points.shape[0],
            inputs=[imgs, points, p_to_i, self.dmax, self.hfov, self.vfov, self.outside, self.safe_ball_size, self.is_depth, self.is_spherical],
            outputs=[col_labels],
            device=self.device
        )

        return wp.to_torch(col_labels)
