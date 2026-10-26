"""Builds models and restores them from checkpoints (used by evaluation and figure scripts)."""
import torch

from .models import (VAE, Autoencoder, Decoder, MLPClassifier, SparseAutoencoder,
                     build_ddpm, build_gan, gan_sample)
from .utils import load_ckpt

DISPLAY = {"ae": "Vanilla AE", "sae": "Sparse AE", "vae": "VAE",
           "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM", "mlp": "MLP Classifier"}


def build(name):
    if name == "mlp":
        return MLPClassifier()
    if name == "ae":
        return Autoencoder()
    if name == "sae":
        return SparseAutoencoder()
    if name == "vae":
        return VAE()
    if name == "frozen_dec":
        return Decoder()
    if name in ("gan", "dcgan"):
        return build_gan(name)[0]          # generator only
    if name == "ddpm":
        return build_ddpm()
    raise ValueError(name)


def load(name, device, strict=True):
    """Returns the model in eval mode, or None if its checkpoint does not exist.
    GANs return the generator; DDPM returns the diffusion wrapper with EMA weights."""
    ck = load_ckpt(name, device)
    if ck is None:
        if strict:
            raise FileNotFoundError(f"checkpoint '{name}' not found - run: python -m src.train --model {name}")
        return None
    m = build(name)
    key = {"gan": "G", "dcgan": "G", "ddpm": "ema"}.get(name, "model")
    m.load_state_dict(ck[key])
    return m.to(device).eval()


def sampler(name, model, device):
    """Returns f(k) -> k generated images in [0, 1] for the generative models."""
    if name == "vae":
        return lambda k: model.sample(k, device)
    if name in ("gan", "dcgan"):
        return lambda k: gan_sample(model, k, device)
    if name == "ddpm":
        return lambda k: model.sample(k, device)
    raise ValueError(f"{name} has no prior sampler")
