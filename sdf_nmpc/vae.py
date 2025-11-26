import os
import torch
from . import default_data_dir
from .utils import preprocessing


class VaeWrapper:
    def __init__(self, cfg):
        self.cfg = cfg

        self.vae = torch.jit.load(os.path.join(default_data_dir(), self.cfg.nn.vae_weights))
        self.vae.to(self.cfg.nn.vae_device)
        self.vae.eval()

        self.preprocess_img = torch.nn.Sequential(
            preprocessing.ToDevice(self.cfg.nn.vae_device),
            torch.jit.script(preprocessing.Reshape(self.cfg.sensor.shape_imgs)),
            torch.jit.script(preprocessing.ClipDistance(self.cfg.sensor.dmax, self.cfg.sensor.mm_resolution)) \
                if not self.cfg.sensor.is_normalized \
                else torch.nn.Identity(),
            torch.jit.script(preprocessing.Depth2Range(self.cfg.sensor.shape_imgs, self.cfg.sensor.hfov, self.cfg.sensor.vfov, self.cfg.nn.vae_device)) \
                if self.cfg.sensor.is_depth \
                else torch.nn.Identity(),
        )
        self.preprocess_latent = preprocessing.ToDevice(self.cfg.nn.vae_device)

        self.img = None
        self.latent = None
        self.decoded = None

    def set_img(self, img):
        self.img = self.preprocess_img(img)

    def set_latent(self, latent):
        self.latent = self.preprocess_latent(latent)

    def encode(self):
        with torch.no_grad():
            self.latent = self.vae.encoder(self.img)
        return self.latent.cpu().numpy()

    def decode(self):
        with torch.no_grad():
            self.decoded = self.vae.decoder(self.latent).reshape(self.cfg.sensor.shape_imgs[1:])
        return self.decoded.cpu().numpy()
