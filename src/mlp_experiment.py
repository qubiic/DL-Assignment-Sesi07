"""Bagian 2 - why an MLP classifier cannot generate images (Tahap 1-5).

Requires checkpoints: mlp (Tahap 1), ae (baseline). vae is optional (reference row in Tahap 5).

    python -m src.mlp_experiment            # full run
    python -m src.mlp_experiment --no-fid   # skip Inception FID in Tahap 5

Outputs:
    results/mlp_experiment.json           all numbers for Bagian III and Q5
    results/mlp_experiment_artifacts.pt   tensors used by Figur 3
    checkpoints/frozen_dec.pt             decoder trained on frozen z_MLP (Tahap 4)
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from . import config as C
from . import zoo
from .data import get_loaders, get_splits
from .metrics import FIDScorer, laplacian_variance, mode_coverage, reconstruction_metrics
from .models import Decoder
from .utils import count_params, get_device, load_ckpt, save_ckpt, save_json, set_seed


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def collect(fn, x, bs=1024):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def weight_analysis(clf, x_test, device):
    """Properties of W1 (256x784), W2 (64x256), W3 (10x64) that block inversion."""
    W = [w.detach().double().cpu() for w in clf.weights]
    with torch.no_grad():
        z, h1 = clf.features(x_test.to(device), return_hidden=True)
        pre1 = clf.fc1(x_test.to(device).flatten(1))
        pre2 = clf.fc2(h1)
    inputs = [x_test.flatten(1).double(), h1.double().cpu(), z.double().cpu()]
    out = {}
    for i, (Wl, a) in enumerate(zip(W, inputs), start=1):
        n_out, n_in = Wl.shape
        sv = torch.linalg.svdvals(Wl)
        rank = int(torch.linalg.matrix_rank(Wl).item())
        WtW = Wl.T @ Wl
        lin = a @ Wl.T                                   # W a  (bias excluded: pure linear map)
        rec_T = lin @ Wl                                 # W^T W a
        rec_pinv = lin @ torch.linalg.pinv(Wl).T         # W^+ W a  (best linear inverse)
        rel = lambda r: ((r - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).mean().item()
        out[f"W{i}"] = {
            "shape_out_in": [n_out, n_in], "rank": rank, "null_space_dim": n_in - rank,
            "sigma_max": sv.max().item(), "sigma_min": sv.min().item(),
            "condition_number": (sv.max() / sv.min()).item(),
            "orth_dev_WtW": ((WtW - torch.eye(n_in, dtype=WtW.dtype)).norm() / n_in ** 0.5).item(),
            "frobenius_WtW_minus_I": (WtW - torch.eye(n_in, dtype=WtW.dtype)).norm().item(),
            "rel_err_transpose_inverse": rel(rec_T),
            "rel_err_pseudo_inverse": rel(rec_pinv),
        }
    out["relu_loss"] = {
        "h1_frac_zeroed": (pre1 <= 0).float().mean().item(),
        "z_frac_zeroed": (pre2 <= 0).float().mean().item(),
        "h1_dead_units": int(((h1 > 0).sum(0) == 0).sum().item()),
        "z_dead_units": int(((z > 0).sum(0) == 0).sum().item()),
    }
    return out


def latent_geometry(z, y):
    """Class structure of a latent space: Fisher ratio, within-class share, effective dim."""
    z = z.double().cpu()
    mu = z.mean(0)
    tot = ((z - mu) ** 2).sum(1).mean().item()
    within = sum(((z[y == c] - z[y == c].mean(0)) ** 2).sum().item() for c in range(10)) / len(z)
    between = tot - within
    ev = torch.linalg.eigvalsh(torch.cov(z.T)).clamp_min(0)
    return {"fisher_ratio": between / within, "within_class_var_share": within / tot,
            "participation_ratio": (ev.sum() ** 2 / (ev ** 2).sum()).item(),
            "frac_exact_zero": (z == 0).double().mean().item()}


def intra_class_retention(x_rec, x, y):
    """sum_c tr Cov(x_rec | c) / sum_c tr Cov(x | c): 1.0 = intra-class variation preserved."""
    num = den = 0.0
    for c in range(10):
        m = y == c
        num += x_rec[m].flatten(1).var(0).sum().item()
        den += x[m].flatten(1).var(0).sum().item()
    return num / den


def train_frozen_decoder(clf, device, epochs):
    """Tahap 4: decoder 64 -> 256 -> 784 on frozen z_MLP, MSE loss (same decoder/loss/epochs as Vanilla AE)."""
    for p in clf.parameters():
        p.requires_grad = False
    clf.eval()
    train, val, _ = get_loaders()
    dec = Decoder().to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=C.LR["frozen_dec"])
    hist, best = [], float("inf")
    for ep in range(1, epochs + 1):
        dec.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, _ in train:
            x = x.to(device)
            with torch.no_grad():
                z = clf.features(x)
            loss = F.mse_loss(dec(z), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        dt = time.perf_counter() - t0
        dec.eval()
        with torch.no_grad():
            vl = sum(F.mse_loss(dec(clf.features(x.to(device))), x.to(device)).item() * len(x)
                     for x, _ in val) / len(val.dataset)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl, "epoch_time_s": dt})
        print(f"[frozen_dec] ep {ep:3d} train {tot / n:.5f} val {vl:.5f} ({dt:.1f}s)")
        if vl < best:
            best = vl
            save_ckpt("frozen_dec", model=dec.state_dict(), epoch=ep, val_loss=vl)
    dec.load_state_dict(load_ckpt("frozen_dec", device)["model"])
    save_json({"model": "frozen_dec", "params": count_params(dec), "epochs": len(hist),
               "avg_epoch_time_s": sum(h["epoch_time_s"] for h in hist) / len(hist),
               "best_val_loss": best, "history": hist},
              os.path.join(C.LOG_DIR, "frozen_dec_history.json"))
    return dec.eval()


def pick_one_per_class(y, seed=C.SEED):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in range(10):
        cand = (y == c).nonzero().flatten()
        idx.append(cand[torch.randint(len(cand), (1,), generator=g)].item())
    return torch.tensor(idx)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--retrain-decoder", action="store_true",
                    help="retrain the Tahap 4 decoder even if checkpoints/frozen_dec.pt exists")
    args = ap.parse_args()
    set_seed(C.SEED)
    device = get_device()
    _, _, test_loader = get_loaders()
    x_test, y_test = get_splits()["test"]

    clf = zoo.load("mlp", device)
    ae = zoo.load("ae", device)
    vae = zoo.load("vae", device, strict=False)
    res = {}

    # Tahap 1 ---------------------------------------------------------
    mlp_ck = load_ckpt("mlp", device)
    res["tahap1"] = {"architecture": "784-256-64-10 (ReLU, dropout 0.2)",
                     "test_acc": mlp_ck.get("test_acc"), "val_acc": mlp_ck.get("val_acc"),
                     "target": C.MLP_TARGET_ACC,
                     "meets_target": (mlp_ck.get("test_acc") or 0) >= C.MLP_TARGET_ACC}
    print(f"[tahap1] test acc {res['tahap1']['test_acc']}")
    res["weights"] = weight_analysis(clf, x_test, device)
    print("[weights]", {k: round(v["orth_dev_WtW"], 3) for k, v in res["weights"].items() if k.startswith("W")})

    # Tahap 2: class one-hot -> pixels via W^T --------------------------
    proj_class = clf.project_class(torch.eye(10, device=device)).cpu()
    class_means = torch.stack([x_test[y_test == c].mean(0) for c in range(10)])
    pc = proj_class.flatten(1)
    cm = class_means.flatten(1)
    corr = torch.corrcoef(torch.cat([pc, cm]))[:10, 10:]      # corr(proj_c, mean_k)
    res["tahap2"] = {
        "corr_with_own_class_mean": corr.diag().tolist(),
        "mean_corr_own_class": corr.diag().mean().item(),
        "mean_corr_other_classes": ((corr.sum() - corr.diag().sum()) / 90).item(),
        "pixel_range": [proj_class.min().item(), proj_class.max().item()],
        "pixel_std_mean": pc.std(1).mean().item(),
        "mse_vs_class_mean": F.mse_loss(proj_class, class_means).item(),
    }

    # Tahap 3: z_MLP of real test images -> pixels via W^T ---------------
    res["tahap3"] = reconstruction_metrics(lambda x: clf.project_latent(clf.features(x)),
                                           test_loader, device)

    # Tahap 4: frozen latent decoding vs Vanilla AE ----------------------
    dec = zoo.load("frozen_dec", device, strict=False)
    if dec is None or args.retrain_decoder:
        dec = train_frozen_decoder(clf, device, C.EPOCHS["frozen_dec"])
    for p in clf.parameters():
        p.requires_grad = False
    frozen_fn = lambda x: dec(clf.features(x))
    m_frozen = reconstruction_metrics(frozen_fn, test_loader, device)
    m_ae = reconstruction_metrics(ae.reconstruct, test_loader, device)
    rec_frozen = collect(lambda x: frozen_fn(x.to(device)).cpu(), x_test)
    rec_ae = collect(lambda x: ae.reconstruct(x.to(device)).cpu(), x_test)
    z_mlp = collect(lambda x: clf.features(x.to(device)).cpu(), x_test)
    z_ae = collect(lambda x: ae.encode(x.to(device)).cpu(), x_test)
    res["tahap4"] = {
        "frozen_mlp_decoder": m_frozen, "vanilla_ae": m_ae,
        "intra_class_retention": {"frozen_mlp_decoder": intra_class_retention(rec_frozen, x_test, y_test),
                                  "vanilla_ae": intra_class_retention(rec_ae, x_test, y_test)},
        "sharpness_laplacian_var": {"real": laplacian_variance(x_test),
                                    "frozen_mlp_decoder": laplacian_variance(rec_frozen),
                                    "vanilla_ae": laplacian_variance(rec_ae)},
        "latent_geometry": {"z_mlp": latent_geometry(z_mlp, y_test),
                            "z_ae": latent_geometry(z_ae, y_test)},
    }
    print(f"[tahap4] PSNR frozen {m_frozen['test_psnr']:.2f} dB vs AE {m_ae['test_psnr']:.2f} dB")

    # Tahap 5: Gaussian sampling in z_MLP (and z_AE for contrast) --------
    mu, sd = z_mlp.mean(0), z_mlp.std(0)
    mu_ae, sd_ae = z_ae.mean(0), z_ae.std(0)
    g = torch.Generator().manual_seed(C.SEED)

    def mlp_prior(k):
        z = mu + sd * torch.randn(k, mu.numel(), generator=g)
        return dec(z.to(device))

    def ae_prior(k):
        z = mu_ae + sd_ae * torch.randn(k, mu_ae.numel(), generator=g)
        return ae.decode(z.to(device))

    with torch.no_grad():
        n_eval = min(C.FID_N, 2000)
        s_mlp = torch.cat([mlp_prior(500).cpu() for _ in range(max(1, n_eval // 500))])
        s_ae = torch.cat([ae_prior(500).cpu() for _ in range(max(1, n_eval // 500))])
        z_rand = mu + sd * torch.randn(10000, mu.numel(), generator=torch.Generator().manual_seed(1))
    res["tahap5"] = {
        "mu_z": mu.tolist(), "sigma_z": sd.tolist(),
        "frac_negative_in_z_rand": (z_rand < 0).float().mean().item(),   # impossible after ReLU
        "frac_zero_in_real_z_mlp": (z_mlp == 0).float().mean().item(),
        "mlp_latent_samples": {**mode_coverage(s_mlp, clf, device), "sharpness": laplacian_variance(s_mlp)},
        "ae_latent_samples": {**mode_coverage(s_ae, clf, device), "sharpness": laplacian_variance(s_ae)},
        "real_test_reference": {**mode_coverage(x_test[:n_eval], clf, device),
                                "sharpness": laplacian_variance(x_test[:n_eval])},
    }
    if vae is not None:
        with torch.no_grad():
            s_vae = torch.cat([vae.sample(500, device).cpu() for _ in range(max(1, n_eval // 500))])
        res["tahap5"]["vae_prior_samples"] = {**mode_coverage(s_vae, clf, device),
                                              "sharpness": laplacian_variance(s_vae)}
    if not args.no_fid:
        try:
            fid = FIDScorer(x_test, device)
            g = torch.Generator().manual_seed(C.SEED)
            res["tahap5"]["fid"] = {"mlp_latent_gaussian": fid.score(mlp_prior),
                                    "ae_latent_gaussian": fid.score(ae_prior)}
            if vae is not None:
                res["tahap5"]["fid"]["vae_prior"] = fid.score(lambda k: vae.sample(k, device))
            print("[tahap5] FID", res["tahap5"]["fid"])
        except Exception as e:
            print(f"[tahap5] FID skipped: {e}")

    # artifacts for Figur 3 ---------------------------------------------
    idx = pick_one_per_class(y_test)
    xs = x_test[idx].to(device)
    with torch.no_grad():
        art = {
            "gt": xs.cpu(), "labels": y_test[idx],
            "proj_class": proj_class,
            "proj_latent": clf.project_latent(clf.features(xs)).cpu(),
            "frozen_recon": frozen_fn(xs).cpu(),
            "ae_recon": ae.reconstruct(xs).cpu(),
            "prior_samples": s_mlp[:20],
        }
    torch.save(art, os.path.join(C.RESULTS_DIR, "mlp_experiment_artifacts.pt"))
    save_json(res, os.path.join(C.RESULTS_DIR, "mlp_experiment.json"))
    print("saved results/mlp_experiment.json")


if __name__ == "__main__":
    main()
