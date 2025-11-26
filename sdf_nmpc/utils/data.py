import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2
from . import preprocessing
import h5py


class ImageAugmenter(torch.nn.Module):
    """Augmentation of images for training.
    shape       -- shape of images
    device      -- 'cpu' or 'cuda'
    noise       -- boolean for augmentation (additive Gaussian noising)
    flip        -- boolean for augmentation (horz/vert flips)
    rotate      -- boolean for augmentation (rotations)
    erase       -- boolean for augmentation (pixel erasing)
    """
    def __init__(self, shape, device, noise=False, flip=False, translate=False, rotate=False, erase=False, outlier_rm=False):
        super(ImageAugmenter, self).__init__()
        self.shape = shape
        self.device = device
        self.noise = noise
        self.flip = flip
        self.translate = translate
        self.rotate = rotate
        self.erase = erase
        self.outlier_rm = outlier_rm
        self.invalid = 0.  # value for invalid pixels

        ## augment params
        self.proba_noise = 1.  # probability of adding Gaussian noise
        self.proba_flip = 0.5  # probability of flipping in each axis
        self.proba_translate = 1  # probability of translating the image along the horizontal axis (for 360° sensor)
        self.proba_rotate = 0.8  # probability of rotating
        self.proba_erase_pixels = 0.3  # probability of erasing random pixels
        self.proba_erase_boxes = 0.3  # probability of erasing random boxes of pixels
        self.std_range = 0.02  # [m] normalized to [0,1]
        self.max_rot = 5  # [degrees]
        ratio_erase_min = 0.03
        ratio_erase_max = 0.10
        self.nb_pix_erase_min = int(self.shape[1] * self.shape[2] * ratio_erase_min)
        self.nb_pix_erase_max = int(self.shape[1] * self.shape[2] * ratio_erase_max)
        self.nb_box_erase_min = 1
        self.nb_box_erase_max = 4
        self.boxes_scale_range = (0.02,0.06)
        self.boxes_ratio_range = (0.2, 5)

        ## augmentations
        self.flipper = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.RandomHorizontalFlip(p=self.proba_flip),
            torchvision.transforms.v2.RandomVerticalFlip(p=self.proba_flip)
        ])
        def _translate(img):
            n = torch.randint(0, self.shape[2], (1,)).item()
            return torch.cat([img[..., n:], img[..., :n]], dim=-1)
        self.translater = torchvision.transforms.v2.RandomApply(
            [torchvision.transforms.v2.Lambda(_translate)],
            p=self.proba_translate
        )
        self.rotater = torchvision.transforms.v2.RandomApply(
            [torchvision.transforms.RandomRotation(degrees=self.max_rot, fill=self.invalid)],
            p=self.proba_rotate
        )
        self.noiser = torchvision.transforms.v2.RandomApply(
            [torchvision.transforms.v2.Lambda(lambda img: torch.where(img != self.invalid, (img + torch.randn_like(img, device=self.device)*self.std_range).clamp(0,1), self.invalid))],
            p=self.proba_noise
        )
        self.eraser_box = torchvision.transforms.RandomErasing(p=1., scale=self.boxes_scale_range, ratio=self.boxes_ratio_range, value=self.invalid, inplace=True)

        self.outlier_remover = preprocessing.RemoveCloseOutliers(kernel_size=3, min_range=0.1, device=self.device)


    def forward(self, img):
        ## check if image has invalid pixels (ie is from real sensor) and outlier rejection should be performed
        apply_outlier_rm = self.outlier_rm and torch.any(img == self.invalid)

        if self.flip:
            img = self.flipper(img)

        if self.translate:
            img = self.translater(img)

        if self.rotate:
            img = self.rotater(img)

        if apply_outlier_rm:
            img_label = self.outlier_remover(img)
        else:
            img_label = torch.clone(img)

        if self.noise:
            img = self.noiser(img)

        if self.erase: # and not apply_outlier_rm:
            ## randomily delete some random pixels and random rectangles in the images
            if torch.rand((1)).item() < self.proba_erase_pixels:
                nb_pix_erase = torch.randint(self.nb_pix_erase_min, self.nb_pix_erase_max, (1,)).item()
                mask_x = torch.randint(0, self.shape[2], (nb_pix_erase,))
                mask_y = torch.randint(0, self.shape[1], (nb_pix_erase,))
                img[:, mask_y, mask_x] = 0
            if torch.rand((1)).item() < self.proba_erase_boxes:
                nb_box_erase = torch.randint(self.nb_box_erase_min, self.nb_box_erase_max+1, (1,)).item()
                for _ in range(nb_box_erase):
                    self.eraser_box(img)

        return img, img_label



class ImageDataset(Dataset):
    """Dataset of images.
    data        -- data h5py handler, size (nb_images,C,H,W)
    idx         -- indices of relevant images in data (for train/valid separation)
    preprocess  -- preprocessing module
    augment     -- ImageAugmenter object
    col_mapping -- collision mapping applied to the label image (eg, DCE, erosion, min_pool...)
    """
    def __init__(self, data, idx, preprocess, augment=None, col_mapping=None):
        self.imgs = data
        self.idx = idx
        self.preprocess = preprocess
        self.augment = augment
        self.augment_idx = idx
        self.col_mapping = col_mapping


    def set_augment_idx(self, augment_idx):
        self.augment_idx = augment_idx


    def __len__(self):
        return len(self.idx)


    def __getitem__(self, idx):
        img_np = self.imgs[self.idx[idx],:,:,:].astype(np.float32)
        img = self.preprocess(img_np)

        if self.augment is not None and idx in self.augment_idx:
            img, img_label = self.augment(img)
        else:
            img_label = img.clone()

        if self.col_mapping is not None:
            img_label = self.col_mapping(img_label)
        img_label = torch.where(img > 0, img_label, 0)
        return img, img_label



def _prepare_dataset(h5file, train, dmax, vae, col_map, device):
    data = h5file['train' if train else 'test']['images']
    metadata = {
        'dmax': dmax,
        'hfov': h5file.attrs['hfov'],
        'vfov': h5file.attrs['vfov'],
        'aspect_ratio': h5file.attrs['aspect_ratio'],
        'is_spherical': h5file.attrs['is_spherical'],
        'is_depth': False,  # is converted to range anyway
        'nb_imgs': data.shape[0],
        'shape_imgs': list(data.shape[1:]),
    }

    ## augmentation
    if vae:
        augment = ImageAugmenter(metadata['shape_imgs'], device, noise=True, flip=True, translate=True, rotate=True, erase=True, outlier_rm=True)
    else:
        augment = ImageAugmenter(metadata['shape_imgs'], device, noise=True, flip=True, translate=True, rotate=False, erase=True, outlier_rm=False)

    ## preprocessing
    preprocess = torch.nn.Sequential(
        preprocessing.ToDevice(device),
        torch.jit.script(preprocessing.ClipDistance(dmax, mm_resolution=1)),
        torch.jit.script(preprocessing.Depth2Range(metadata['shape_imgs'], metadata['hfov'], metadata['vfov'], device)) \
            if h5file.attrs['is_depth'] else torch.nn.Identity()
    )

    ## collision mapping (erosion)
    if col_map:
        kernel_size = 10
        kernel = np.fromfunction(lambda x, y: ((x-kernel_size)**2 + (y-kernel_size)**2 <= kernel_size**2)*1, (2*kernel_size+1, 2*kernel_size+1), dtype=int).astype(np.uint8)  # circle
        # kernel = np.ones((kernel_size, kernel_size))  # square
        col_mapping = preprocessing.Erode(kernel, True, device)
    else:
        col_mapping = None

    return data, metadata, preprocess, augment, col_mapping


def test_dataset_from_h5(path_to_data, dataset, dmax, batch_size=1, shuffle=False, vae=False, col_map=False, device='cuda'):
    """Parse the hdf5 file and returns the corresponding DataLoader and metadata dict.
    """
    h5file = h5py.File(os.path.join(path_to_data, dataset), 'r')
    data, metadata, preprocess, augment, col_mapping = _prepare_dataset(h5file, False, dmax, vae, col_map, device)
    augment = None
    # augment = ImageAugmenter(metadata['shape_imgs'], device, noise=True, flip=True, translate=True, rotate=True, erase=True, outlier_rm=True)
    test_dataloader = DataLoader(ImageDataset(data, range(metadata['nb_imgs']), preprocess=preprocess, augment=augment, col_mapping=col_mapping), batch_size=1, shuffle=shuffle)

    return test_dataloader, metadata


def train_dataset_from_h5(path_to_data, dataset, dmax, batch_size_train=1, batch_size_valid=None, train_valid_ratio=0.8, vae=False, col_map=False, seed=42, device='cuda'):
    """Parse the hdf5 file and returns the corresponding DataLoader and metadata dict.
    If batch_size_valid is None, it is set to batch_size_train.
    If train_valid_ratio is 1, the valid_dataloader will be None.
    Returns a tuple of (train, valid) dataloaders.
    """
    h5file = h5py.File(os.path.join(path_to_data, dataset), 'r')
    data, metadata, preprocess, augment, col_mapping = _prepare_dataset(h5file, True, dmax, vae, col_map, device)
    if not batch_size_valid: batch_size_valid = batch_size_train

    dataset = ImageDataset(data, range(metadata['nb_imgs']), preprocess=preprocess, augment=augment, col_mapping=col_mapping)

    ## splitting seed is kept constant for repetability and thus allows to resume training
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_valid_ratio, 1-train_valid_ratio], generator=torch.Generator().manual_seed(seed))
    dataset.set_augment_idx(train_dataset.indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    return (train_dataloader, valid_dataloader), metadata
