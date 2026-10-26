"""Evaluation metrics: MSE / PSNR / SSIM, FID, inference latency, sharpness, mode coverage."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


# ------------------------------------------------------------ reconstruction
@torch.no_grad()
def reconstruction_metrics(recon_fn, loader, device):
    """Per-image MSE, PSNR (data range 1) and SSIM averaged over the whole loader."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    mse, psnr, ssim = [], [], []
    for x, _ in loader:
        x = x.to(device)
        x_hat = recon_fn(x).clamp(0, 1)
        m = F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(1)
        mse.append(m.cpu())
        psnr.append((10 * torch.log10(1.0 / m.clamp_min(1e-10))).cpu())
        ssim.append(ssim_fn(x_hat, x, data_range=1.0, reduction="none").cpu())
    mse, psnr, ssim = (torch.cat(v) for v in (mse, psnr, ssim))
    return {"test_mse": mse.mean().item(), "test_mse_std": mse.std().item(),
            "test_psnr": psnr.mean().item(), "test_psnr_std": psnr.std().item(),
            "test_ssim": ssim.mean().item(), "test_ssim_std": ssim.std().item()}


# ------------------------------------------------------------ FID
class FIDScorer:
    """Inception-v3 FID (2048-d pool features). Grayscale is repeated to 3 channels and
    resized to 299x299 inside torchmetrics. Real statistics are computed once from the
    test split and reused for every model."""

    def __init__(self, real_images, device, batch=256):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device = device
        self.batch = batch
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                            reset_real_features=False).to(device)
        for i in range(0, len(real_images), batch):
            self.fid.update(self._rgb(real_images[i:i + batch]), real=True)
        self.n_real = len(real_images)

    def _rgb(self, x):
        return x.to(self.device).clamp(0, 1).repeat(1, 3, 1, 1)

    @torch.no_grad()
    def score(self, sample_fn, n=C.FID_N, batch=None):
        """sample_fn(k) -> k images in [0, 1]."""
        batch = batch or self.batch
        self.fid.reset()                     # keeps real features
        done = 0
        while done < n:
            k = min(batch, n - done)
            self.fid.update(self._rgb(sample_fn(k)), real=False)
            done += k
        return self.fid.compute().item()

    def score_images(self, images):
        return self.score(_Feeder(images), n=len(images))


class _Feeder:
    def __init__(self, images):
        self.images, self.i = images, 0

    def __call__(self, k):
        out = self.images[self.i:self.i + k]
        self.i += k
        return out


# ------------------------------------------------------------ latency
def measure_latency(fn, device, reps=C.LATENCY_REPS, warmup=C.LATENCY_WARMUP):
    """Wall-clock latency (ms) of fn() which produces one batch of 64 images."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)), "reps": reps}


# ------------------------------------------------------------ sharpness (Q2)
_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


@torch.no_grad()
def laplacian_variance(x):
    """Mean variance of the Laplacian response; higher = sharper edges."""
    lap = F.conv2d(x.float().cpu(), _LAPLACE, padding=1)
    return lap.flatten(1).var(1).mean().item()


# ------------------------------------------------------------ mode coverage (Q3 trilemma)
@torch.no_grad()
def mode_coverage(images, classifier, device, n_classes=10):
    """Class histogram of generated samples according to the trained MLP classifier.
    entropy_ratio = H(p_gen) / log(10) (1.0 = perfectly balanced); kl_uniform = KL(p_gen || U)."""
    classifier.eval()
    probs = []
    for i in range(0, len(images), 512):
        probs.append(torch.softmax(classifier(images[i:i + 512].to(device)), 1).cpu())
    probs = torch.cat(probs)
    hist = torch.bincount(probs.argmax(1), minlength=n_classes).float()
    p = hist / hist.sum()
    nz = p[p > 0]
    entropy = max(0.0, -(nz * nz.log()).sum().item())
    return {"class_hist": p.tolist(),
            "entropy_ratio": entropy / math.log(n_classes),
            "kl_uniform": math.log(n_classes) - entropy,
            "classes_covered": int((p >= 0.02).sum().item()),
            "mean_confidence": probs.max(1).values.mean().item()}
