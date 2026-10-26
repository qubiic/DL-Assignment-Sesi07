"""Deterministic autoencoders (Sesi 06).

Encoder 784 -> 256 -> 64 deliberately mirrors the MLP classifier's feature
extractor, and the decoder 64 -> 256 -> 784 is the same module that is trained
on frozen z_MLP in the Bagian 2 experiment. The only difference between the
two pipelines is therefore the objective the encoder was trained with.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


class Decoder(nn.Module):
    def __init__(self, latent_dim=C.LATENT_AE, hidden=C.HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 784), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, *C.IMG_SHAPE)


class Autoencoder(nn.Module):
    """Vanilla AE: L = MSE(x, x_hat)."""

    def __init__(self, latent_dim=C.LATENT_AE, hidden=C.HIDDEN):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, latent_dim)      # linear bottleneck
        self.decoder = Decoder(latent_dim, hidden)

    def encode(self, x, return_hidden=False):
        h = F.relu(self.fc1(x.flatten(1)))
        z = self.fc2(h)
        return (z, h) if return_hidden else z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, h = self.encode(x, return_hidden=True)
        return self.decode(z), z, h

    def compute_loss(self, x):
        x_hat, _, h = self(x)
        rec = F.mse_loss(x_hat, x)
        return rec, {"recon": rec.item()}

    @torch.no_grad()
    def reconstruct(self, x):
        return self(x)[0]


class SparseAutoencoder(Autoencoder):
    """Sparse AE: L = MSE(x, x_hat) + lambda * mean|h|, h = 256-d ReLU hidden code."""

    def __init__(self, latent_dim=C.LATENT_AE, hidden=C.HIDDEN, sparse_lambda=C.SPARSE_LAMBDA):
        super().__init__(latent_dim, hidden)
        self.sparse_lambda = sparse_lambda

    def compute_loss(self, x):
        x_hat, _, h = self(x)
        rec = F.mse_loss(x_hat, x)
        l1 = h.abs().sum(1).mean() / h.shape[1]
        loss = rec + self.sparse_lambda * l1
        return loss, {"recon": rec.item(), "l1": l1.item(),
                      "active_frac": (h > 1e-3).float().mean().item()}
