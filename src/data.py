"""Fashion-MNIST loading with the standardized 80/10/10 split.

The official train (60k) and test (10k) sets are pooled into 70,000 images and
re-split with a fixed permutation (seed 42), so every model sees exactly the
same train / val / test indices.
"""
import gzip
import os
import shutil
import urllib.request

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from . import config as C

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_GITHUB_MIRROR = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
          "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]


def _fetch_from_github(root):
    """Fallback when the default torchvision mirrors are unreachable."""
    raw = os.path.join(root, "FashionMNIST", "raw")
    os.makedirs(raw, exist_ok=True)
    for name in _FILES:
        out = os.path.join(raw, name)
        if os.path.exists(out):
            continue
        gz = out + ".gz"
        urllib.request.urlretrieve(_GITHUB_MIRROR + name + ".gz", gz)
        with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def _load_official(root):
    try:
        tr = datasets.FashionMNIST(root, train=True, download=True)
        te = datasets.FashionMNIST(root, train=False, download=True)
    except Exception as e:  # mirror blocked / offline
        print(f"[data] torchvision download failed ({e.__class__.__name__}), using GitHub mirror")
        _fetch_from_github(root)
        tr = datasets.FashionMNIST(root, train=True, download=False)
        te = datasets.FashionMNIST(root, train=False, download=False)
    x = torch.cat([tr.data, te.data]).unsqueeze(1).float() / 255.0   # (70000, 1, 28, 28) in [0, 1]
    y = torch.cat([tr.targets, te.targets]).long()
    return x, y


_CACHE = {}


def get_splits(seed: int = C.SEED):
    """Returns dict(train=(x, y), val=(x, y), test=(x, y)); pixels in [0, 1]."""
    if seed in _CACHE:
        return _CACHE[seed]
    x, y = _load_official(C.DATA_DIR)
    n = len(x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_tr, n_va = int(C.SPLIT[0] * n), int(C.SPLIT[1] * n)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    if C.QUICK:
        idx = {"train": idx["train"][:2048], "val": idx["val"][:512], "test": idx["test"][:512]}
    splits = {k: (x[v], y[v]) for k, v in idx.items()}
    _CACHE[seed] = splits
    return splits


def get_loaders(batch_size: int = C.BATCH_SIZE, seed: int = C.SEED):
    s = get_splits(seed)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(*s["train"]), batch_size=batch_size, shuffle=True,
                       generator=g, drop_last=False)
    val = DataLoader(TensorDataset(*s["val"]), batch_size=batch_size, shuffle=False)
    test = DataLoader(TensorDataset(*s["test"]), batch_size=batch_size, shuffle=False)
    return train, val, test


if __name__ == "__main__":
    s = get_splits()
    for k, (x, y) in s.items():
        print(f"{k:5s} x={tuple(x.shape)} y={tuple(y.shape)} "
              f"class counts={torch.bincount(y, minlength=10).tolist()}")
