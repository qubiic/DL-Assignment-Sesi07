import json
import os
import random
import time

import numpy as np
import torch

from . import config as C


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def ckpt_path(name: str) -> str:
    return os.path.join(C.CKPT_DIR, f"{name}.pt")


def save_ckpt(name: str, **state):
    torch.save(state, ckpt_path(name))


def load_ckpt(name: str, device="cpu"):
    path = ckpt_path(name)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(type(o))


def to_unit(x):
    """[-1, 1] -> [0, 1]"""
    return ((x + 1) / 2).clamp(0, 1)


def to_sym(x):
    """[0, 1] -> [-1, 1]"""
    return x * 2 - 1


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
