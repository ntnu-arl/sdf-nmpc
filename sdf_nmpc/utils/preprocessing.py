import numpy as np
import torch


class Depth2Range(torch.nn.Module):
    """Transform depth to range.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    shape       -- shape of images
    hfov        -- horizontal fov
    vfov        -- vertical fov
    device      -- a torch.device
    """
    def __init__(self, shape, hfov, vfov, device='cpu'):
        super(Depth2Range, self).__init__()
        self.device = device

        height = shape[-2]
        width = shape[-1]

        u = torch.arange(0, width, 1, dtype=torch.float32, device=device)
        v = torch.arange(0, height, 1, dtype=torch.float32, device=device)
        u, v = torch.meshgrid(u, v, indexing='xy')

        tan_hfov = torch.tan(torch.tensor(hfov, dtype=torch.float32, device=device))
        tan_vfov = torch.tan(torch.tensor(vfov, dtype=torch.float32, device=device))

        self.yz_sqrt = torch.sqrt(1 + (tan_hfov * (1 - 2 * u / width))**2 + (tan_vfov * (1 - 2 * v / height))**2)

    def forward(self, depth_img):
        return torch.clip(depth_img * self.yz_sqrt, 0, 1)



class Range2Depth(torch.nn.Module):
    """Transform range to depth.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    shape       -- shape of images
    hfov        -- horizontal fov
    vfov        -- vertical fov
    device      -- a torch.device
    """
    def __init__(self, shape, hfov, vfov, device='cpu'):
        # assert hfov < np.pi / 2, 'depth image not defined for hfov >= pi/2'
        super(Range2Depth, self).__init__()
        self.device = device

        height = shape[-2]
        width = shape[-1]

        u = torch.arange(0, width, 1, dtype=torch.float32, device=device)
        v = torch.arange(0, height, 1, dtype=torch.float32, device=device)
        u, v = torch.meshgrid(u, v, indexing='xy')

        tan_hfov = torch.tan(torch.tensor(hfov, dtype=torch.float32, device=device))
        tan_vfov = torch.tan(torch.tensor(vfov, dtype=torch.float32, device=device))

        y = tan_hfov * (1 - 2 * u / width)
        z = tan_vfov * (1 - 2 * v / height)
        self.norm = torch.sqrt(1 + y**2 + z**2)


    def forward(self, range_img):
        return range_img / self.norm



class ClipDistance(torch.nn.Module):
    """Clip images to a given dmax.
    Returns dmax-normalized float32 tensor.
    dmax            -- max distance in images [m]
    mm_resolution   -- distance resolution of pixel values [mm]
    """
    def __init__(self, dmax, mm_resolution=1000):
        super(ClipDistance, self).__init__()
        self.dmax = dmax / mm_resolution * 1000


    def forward(self, img):
        return torch.clip(img / self.dmax, 0, 1)



class Reshape(torch.nn.Module):
    """Reshape and expand image shape to [B, C, H, W] tensor.
    shape_img   -- desire target image size. If None, no resizing is performed.
    """
    def __init__(self, shape_img=None):
        super(Reshape, self).__init__()
        self.shape_img = shape_img


    def forward(self, img):
        img = img.view(1, 1, img.shape[-2], img.shape[-1])
        if self.shape_img is not None and img.shape[-2:] != self.shape_img[-2:]:
            img = torch.nn.functional.interpolate(img, size=self.shape_img[-2:], mode='bilinear')
        return img



class Dilate(torch.nn.Module):
    """Perform dilatation of an image.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    kernel          -- morphological kernel (numpy array of 0s and 1s).
    ignore_zeros    -- ignore 0-valued pixels or not.
    device          -- a torch.device
    """
    def __init__(self, kernel=np.ones((3,3)), ignore_zeros=False, device='cpu'):
        super(Dilate, self).__init__()
        self.device = device
        if not torch.is_tensor(kernel):
            kernel = torch.from_numpy(kernel)
        self.kernel = kernel.to(torch.float32).to(self.device)
        self.ignore_zeros = ignore_zeros
        self.border_val = -2

        ## get padding dimensions
        k_h, k_w = self.kernel.shape
        origin = [k_h // 2, k_w // 2]
        self.pad = [origin[1], k_w - origin[1] - 1, origin[0], k_h - origin[0] - 1]

        ## mask out 0 elements of the kernel
        self.mask = torch.zeros_like(self.kernel)
        self.mask[self.kernel == 0] = self.border_val
        self.mask = self.mask.view(-1)

        ## reshape kernel for handling channels
        kernel = torch.eye(k_h * k_w, dtype=self.kernel.dtype, device=self.device)
        self.kernel_reshaped = kernel.view(k_h * k_w, 1, k_h, k_w)


    def forward(self, img):
        img4d = img.unsqueeze(0) if img.ndim == 3 else img

        if self.ignore_zeros:
            img4d[img4d == 0] = self.border_val
        padded = torch.nn.functional.pad(img4d, self.pad, mode='constant', value=self.border_val)
        B, C, H, W = img4d.shape
        Hpad, Wpad = padded.shape[-2:]
        padded = padded.view(B * C, 1, Hpad, Wpad)
        padded, _ = torch.nn.functional.conv2d(
                padded.view(B * C, 1, Hpad, Wpad), self.kernel_reshaped, padding=0, bias=self.mask.flip(0)
        ).max(dim=1)
        padded = padded.view(B, C, H, W)

        if self.ignore_zeros:
            padded[padded == self.border_val] = 0

        return padded.view_as(img)



class Erode(torch.nn.Module):
    """Perform erosion of an image.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    kernel          -- morphological kernel (numpy array of 0s and 1s).
    ignore_zeros    -- ignore 0-valued pixels or not.
    device          -- a torch.device
    """
    def __init__(self, kernel=np.ones((3,3)), ignore_zeros=False, device='cpu'):
        super(Erode, self).__init__()
        self.device = device
        if not torch.is_tensor(kernel):
            kernel = torch.from_numpy(kernel)
        self.kernel = kernel.to(torch.float32).to(self.device)
        self.ignore_zeros = ignore_zeros
        self.border_val = 2

        ## get padding dimensions
        k_h, k_w = self.kernel.shape
        origin = [k_h // 2, k_w // 2]
        self.pad = [origin[1], k_w - origin[1] - 1, origin[0], k_h - origin[0] - 1]

        ## mask out 0 elements of the kernel
        self.mask = torch.zeros_like(self.kernel)
        self.mask[self.kernel == 0] = self.border_val
        self.mask = -self.mask.view(-1)

        ## reshape kernel for handling channels
        kernel = torch.eye(k_h * k_w, dtype=self.kernel.dtype, device=self.device)
        self.kernel_reshaped = kernel.view(k_h * k_w, 1, k_h, k_w)


    def forward(self, img):
        img4d = img.unsqueeze(0) if img.ndim == 3 else img

        if self.ignore_zeros:
            img4d[img4d == 0] = self.border_val
        padded = torch.nn.functional.pad(img4d, self.pad, mode='constant', value=self.border_val)
        B, C, H, W = img4d.shape
        Hpad, Wpad = padded.shape[-2:]
        padded, _ = torch.nn.functional.conv2d(
                padded.view(B * C, 1, Hpad, Wpad), self.kernel_reshaped, padding=0, bias=-self.mask
        ).min(dim=1)
        padded = padded.view(B, C, H, W)

        if self.ignore_zeros:
            padded[padded == self.border_val] = 0

        return padded.view_as(img)



class Open(torch.nn.Module):
    """Perform opening of an image.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    kernel_erode    -- erosion morphological kernel (numpy array of 0s and 1s).
    kernel_dilate   -- dilatation morphological kernel (numpy array of 0s and 1s).
    device          -- a torch.device
    """
    def __init__(self, kernel_erode=np.ones((3,3)), kernel_dilate=np.ones((3,3)), device='cpu'):
        super(Open, self).__init__()
        self.device = device
        self.erode = Erode(kernel_erode, False, device)
        self.dilate = Dilate(kernel_dilate, False, device)


    def forward(self, img):
        return self.dilate(self.erode(img))



class Close(torch.nn.Module):
    """Perform closing of an image.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    kernel_erode    -- erosion morphological kernel (numpy array of 0s and 1s).
    kernel_dilate   -- dilatation morphological kernel (numpy array of 0s and 1s).
    device          -- a torch.device
    """
    def __init__(self, kernel_erode=np.ones((3,3)), kernel_dilate=np.ones((3,3)), device='cpu'):
        super(Close, self).__init__()
        self.device = device
        self.erode = Erode(kernel_erode, False, device)
        self.dilate = Dilate(kernel_dilate, False, device)


    def forward(self, img):
        return self.erode(self.dilate(img))



class RemoveCloseOutliers(torch.nn.Module):
    """Perform opening on the depth to remove the outliers in the shadow area of real sensors.
    The non-zeros pixels post-opening are assigned the value of the corresponding input pixels.
    Assumes dmax-normalized [C, H, W] or [B, C, H, W] tensor on the correct device.
    device          -- a torch.device
    """
    def __init__(self, kernel_size=3, min_range=0.1, device='cpu'):
        super(RemoveCloseOutliers, self).__init__()
        kernel = np.ones((kernel_size, kernel_size))
        self.open = Open(kernel, kernel, device)
        self.min_range = min_range
        self.device = device


    def forward(self, img):
        img = torch.where(img < self.min_range, 0, img)  # arbitrary crop of short values
        morph = self.open(img)
        morph[morph > 0] = img[morph > 0]
        return morph



class ToDevice(torch.nn.Module):
    """Transform input to torch Tensor of torch.float32 on the specified device.
    device      -- a torch.device
    """
    def __init__(self, device='cpu'):
        super(ToDevice, self).__init__()
        self.device = device


    def forward(self, img):
        if not torch.is_tensor(img):
            return torch.from_numpy(img.astype(np.float32)).to(self.device)
        else:
             return img.to(torch.float32).to(self.device)
