"""Vanilla GAN (fully connected, Goodfellow et al., 2014) and DCGAN (Radford et al., 2016).

Both generators output tanh images in [-1, 1]; data are mapped [0,1] -> [-1,1]
before reaching the discriminator. Discriminators return logits, so the loss is
BCEWithLogits with the non-saturating generator objective.
"""
import torch
import torch.nn as nn

from .. import config as C


class MLPGenerator(nn.Module):
    def __init__(self, z_dim=C.LATENT_GAN):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.flatten(1)).view(-1, *C.IMG_SHAPE)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)


class DCGANGenerator(nn.Module):
    """z (100x1x1) -> 256x7x7 -> 128x14x14 -> 1x28x28. No fully connected layer."""

    def __init__(self, z_dim=C.LATENT_GAN, ngf=64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.shape[0], self.z_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """1x28x28 -> 64x14x14 -> 128x7x7 -> 1 (strided convs, no pooling, no BN on input layer)."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_weights_init(m):
    """N(0, 0.02) init from the DCGAN paper."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build_gan(kind: str):
    if kind == "gan":
        return MLPGenerator(), MLPDiscriminator()
    if kind == "dcgan":
        G, D = DCGANGenerator(), DCGANDiscriminator()
        G.apply(dcgan_weights_init)
        D.apply(dcgan_weights_init)
        return G, D
    raise ValueError(kind)


@torch.no_grad()
def gan_sample(G, n, device, generator=None, z=None):
    """Returns images in [0, 1]."""
    if z is None:
        z = torch.randn(n, G.z_dim, device=device, generator=generator)
    return ((G(z) + 1) / 2).clamp(0, 1)
