"""Variational Autoencoder (Kingma & Welling, 2014).

ELBO with a Bernoulli likelihood:  L = BCE_sum(x, x_hat) + KL(q(z|x) || N(0, I)),
both summed over pixels / latent dims and averaged over the batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from .autoencoder import Decoder


class VAE(nn.Module):
    def __init__(self, latent_dim=C.LATENT_AE, hidden=C.HIDDEN, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.fc1 = nn.Linear(784, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = Decoder(latent_dim, hidden)

    def encode(self, x):
        h = F.relu(self.fc1(x.flatten(1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_loss(self, x):
        x_hat, mu, logvar = self(x)
        b = x.shape[0]
        rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        loss = rec + self.beta * kl
        return loss, {"recon_bce": rec.item(), "kl": kl.item(),
                      "mse": F.mse_loss(x_hat, x).item()}

    @torch.no_grad()
    def reconstruct(self, x):
        mu, _ = self.encode(x)          # deterministic: decode the posterior mean
        return self.decode(mu)

    @torch.no_grad()
    def sample(self, n, device, generator=None):
        z = torch.randn(n, self.latent_dim, device=device, generator=generator)
        return self.decode(z)
