"""Bagian V - the five mandatory figures (300 dpi PNG + PDF in results/figures/).

    python -m src.figures              # all five
    python -m src.figures --only 2 4   # a subset

Figur 3 reads results/mlp_experiment_artifacts.pt (run src.mlp_experiment first).
Figur 4 also writes results/interpolation_metrics.json (Q4).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torchvision.utils import make_grid

from . import config as C
from . import zoo
from .data import CLASSES, get_splits
from .mlp_experiment import pick_one_per_class
from .utils import get_device, load_json, save_json, set_seed

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK = "#1f2933"
PALETTE = {"ae": "#2f6690", "sae": "#3a7d44", "vae": "#9c6644",
           "gan": "#b23a48", "dcgan": "#7b2cbf", "ddpm": "#e09f3e"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG_DIR, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved results/figures/{name}.png")


def img_row(axes, imgs, cmap="gray", norm=False):
    for ax, im in zip(axes, imgs):
        a = im.squeeze().cpu().numpy()
        if norm:
            a = (a - a.min()) / max(1e-8, a.max() - a.min())
        ax.imshow(a, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def row_label(ax, text):
    ax.set_ylabel(text, rotation=0, ha="right", va="center", fontsize=9, color=INK, labelpad=8)


def grid_img(x, nrow):
    return make_grid(x.cpu(), nrow=nrow, padding=2, pad_value=1.0)[0].numpy()


def metrics():
    return load_json(os.path.join(C.RESULTS_DIR, "metrics.json")) or {"models": {}}


# ------------------------------------------------------------------ Figur 1
@torch.no_grad()
def figure1(device):
    x_test, y_test = get_splits()["test"]
    idx = pick_one_per_class(y_test)
    x = x_test[idx].to(device)
    m = metrics()["models"]
    rows = [("Ground truth", x.cpu())]
    for name in ("ae", "sae", "vae"):
        model = zoo.load(name, device)
        r = m.get(name, {})
        lab = zoo.DISPLAY[name]
        if "test_psnr" in r:
            lab += f"\nPSNR {r['test_psnr']:.2f} dB\nSSIM {r['test_ssim']:.3f}"
        rows.append((lab, model.reconstruct(x).cpu()))
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 4.6))
    for i, (lab, imgs) in enumerate(rows):
        img_row(axes[i], imgs)
        row_label(axes[i, 0], lab)
    for j, c in enumerate(y_test[idx].tolist()):
        axes[0, j].set_title(CLASSES[c], fontsize=8)
    fig.suptitle("Figur 1. Rekonstruksi citra uji: Vanilla AE vs Sparse AE vs VAE (Fashion-MNIST)",
                 y=1.0, fontsize=11)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, "fig1_reconstruction_grid")


# ------------------------------------------------------------------ Figur 2
@torch.no_grad()
def figure2(device):
    m = metrics()["models"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.4))
    panels = [("A", "vae", "VAE, z ~ N(0, I)"), ("B", "gan", "Vanilla GAN, z ~ N(0, I)"),
              ("C", "dcgan", "DCGAN, z ~ N(0, I)"), ("D", "ddpm", "DDPM, x_T ~ N(0, I), T = 1000")]
    for ax, (tag, name, title) in zip(axes.flat, panels):
        set_seed(C.SEED)
        model = zoo.load(name, device)
        if name == "ddpm":
            g = torch.Generator(device=device).manual_seed(C.SEED)
            s = model.sample(16, device, generator=g)
        else:
            s = zoo.sampler(name, model, device)(16)
        ax.imshow(grid_img(s, 4), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fid = m.get(name, {}).get("fid")
        ax.set_title(f"Panel {tag}: {title}" + (f"\nFID = {fid:.2f}" if fid is not None else ""), fontsize=9.5)
    fig.suptitle("Figur 2. Sampling generatif acak 4x4 per model", fontsize=11, y=0.995)
    fig.tight_layout()
    save(fig, "fig2_random_sampling")


# ------------------------------------------------------------------ Figur 3
def figure3(device):
    art_path = os.path.join(C.RESULTS_DIR, "mlp_experiment_artifacts.pt")
    if not os.path.exists(art_path):
        print("figure 3 skipped: run python -m src.mlp_experiment first")
        return
    a = torch.load(art_path, weights_only=False)
    exp = load_json(os.path.join(C.RESULTS_DIR, "mlp_experiment.json")) or {}
    t4 = exp.get("tahap4", {})
    lab_frozen = "(3a) Frozen decoder\nD(z_MLP)"
    lab_ae = "(3b) Vanilla AE\nD(z_AE)"
    if t4:
        lab_frozen += f"\nPSNR {t4['frozen_mlp_decoder']['test_psnr']:.2f} dB"
        lab_ae += f"\nPSNR {t4['vanilla_ae']['test_psnr']:.2f} dB"
    lab_t3 = "(2b) Transpose dari\nz_MLP citra uji"
    if "tahap3" in exp:
        lab_t3 += f"\nPSNR {exp['tahap3']['test_psnr']:.2f} dB"
    rows = [
        ("(1) Citra asli", a["gt"], False),
        ("(2a) Transpose dari\nvektor kelas y_c", a["proj_class"], False),
        ("(2a') idem, kontras\ndinormalisasi", a["proj_class"], True),
        (lab_t3, a["proj_latent"], False),
        (lab_frozen, a["frozen_recon"], False),
        (lab_ae, a["ae_recon"], False),
        ("(4) z_rand ~ N(mu_z,\ndiag(sigma_z^2))", a["prior_samples"][:10], False),
    ]
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 8.6))
    for i, (lab, imgs, norm) in enumerate(rows):
        img_row(axes[i], imgs, norm=norm)
        row_label(axes[i, 0], lab)
    for j, c in enumerate(a["labels"].tolist()):
        axes[0, j].set_title(CLASSES[c], fontsize=8)
    fig.suptitle("Figur 3. Bukti empiris batasan bobot dan fitur laten MLP Classifier untuk image generation",
                 fontsize=11, y=0.995)
    fig.subplots_adjust(wspace=0.04, hspace=0.08, top=0.93)
    save(fig, "fig3_mlp_classifier_limits")


# ------------------------------------------------------------------ Figur 4
@torch.no_grad()
def figure4(device, class_a=1, class_b=8):
    x_test, y_test = get_splits()["test"]
    x_train = get_splits()["train"][0][:10000].to(device).flatten(1)
    clf = zoo.load("mlp", device)
    idx = pick_one_per_class(y_test)
    xa, xb = x_test[idx[class_a]:idx[class_a] + 1].to(device), x_test[idx[class_b]:idx[class_b] + 1].to(device)
    alphas_fig = torch.linspace(0, 1, 6)
    alphas_fine = torch.linspace(0, 1, 21)
    rows, stats = [], {}

    def path(decode, za, zb, alphas):
        z = torch.stack([(1 - a) * za + a * zb for a in alphas.tolist()]).to(device)
        return decode(z).clamp(0, 1)

    def path_stats(imgs):
        flat = imgs.flatten(1)
        steps = (flat[1:] - flat[:-1]).norm(dim=1)
        nn = torch.cdist(flat, x_train).min(1).values
        conf = torch.softmax(clf(imgs), 1)
        return {"smoothness_max_over_mean_step": (steps.max() / steps.mean()).item(),
                "nn_dist_interior_over_endpoints": (nn[1:-1].mean() / nn[[0, -1]].mean()).item(),
                "min_classifier_confidence": conf.max(1).values.min().item(),
                "p_classA_path": conf[:, class_a].tolist(), "p_classB_path": conf[:, class_b].tolist()}

    for name in ("ae", "sae", "vae"):
        m = zoo.load(name, device)
        if name == "vae":
            za, zb = m.encode(xa)[0][0], m.encode(xb)[0][0]
        else:
            za, zb = m.encode(xa)[0], m.encode(xb)[0]
        rows.append((zoo.DISPLAY[name], path(m.decode, za, zb, alphas_fig).cpu()))
        stats[name] = path_stats(path(m.decode, za, zb, alphas_fine))

    for name in ("gan", "dcgan"):
        G = zoo.load(name, device)
        z = torch.randn(2000, G.z_dim, device=device, generator=torch.Generator(device=device).manual_seed(C.SEED))
        p = torch.softmax(clf(((G(z) + 1) / 2).clamp(0, 1)), 1)
        za, zb = z[p[:, class_a].argmax()], z[p[:, class_b].argmax()]
        dec = lambda zz, G=G: ((G(zz) + 1) / 2)
        rows.append((zoo.DISPLAY[name] + "\n(z dipilih via\nclassifier)", path(dec, za, zb, alphas_fig).cpu()))
        stats[name] = path_stats(path(dec, za, zb, alphas_fine))

    fig, axes = plt.subplots(len(rows), 6, figsize=(6.6, 6.4))
    for i, (lab, imgs) in enumerate(rows):
        img_row(axes[i], imgs)
        s = stats[["ae", "sae", "vae", "gan", "dcgan"][i]]
        row_label(axes[i, 0], lab)
        axes[i, -1].text(1.08, 0.5, f"NN-ratio {s['nn_dist_interior_over_endpoints']:.2f}\n"
                                    f"min conf {s['min_classifier_confidence']:.2f}",
                         transform=axes[i, -1].transAxes, fontsize=7.5, va="center", color=INK)
    for j, a in enumerate(alphas_fig.tolist()):
        axes[0, j].set_title(f"alpha = {a:.1f}", fontsize=8.5)
    fig.suptitle(f"Figur 4. Interpolasi laten linier {CLASSES[class_a]} -> {CLASSES[class_b]}",
                 fontsize=11, y=1.0)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, top=0.9)
    save(fig, "fig4_latent_interpolation")
    stats["classes"] = [CLASSES[class_a], CLASSES[class_b]]
    stats["notes"] = ("smoothness = max/mean consecutive pixel-L2 step on 21-point path (1 = uniform); "
                      "NN-ratio = mean distance of interior points to nearest of 10k train images divided "
                      "by the same for the endpoints (>1 = path leaves the data manifold); "
                      "min conf = lowest MLP-classifier max-probability along the path.")
    save_json(stats, os.path.join(C.RESULTS_DIR, "interpolation_metrics.json"))


# ------------------------------------------------------------------ Figur 5
def _hist(name):
    return load_json(os.path.join(C.LOG_DIR, f"{name}_history.json"))


@torch.no_grad()
def figure5(device):
    fig = plt.figure(figsize=(11, 10.2))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.55, top=0.93)

    for k, name in enumerate(("ae", "sae", "vae")):
        ax = fig.add_subplot(gs[0, k])
        h = _hist(name)
        if h is None:
            ax.set_title(f"{zoo.DISPLAY[name]} (no log)"); continue
        ep = [e["epoch"] for e in h["history"]]
        ax.plot(ep, [e["train_loss"] for e in h["history"]], color=PALETTE[name], lw=1.6, label="train")
        ax.plot(ep, [e["val_loss"] for e in h["history"]], color=PALETTE[name], lw=1.6, ls="--", label="val")
        ax.set_xlim(left=0)
        unit = "BCE + KL (nats/citra)" if name == "vae" else ("MSE + lambda*L1" if name == "sae" else "MSE")
        ax.set_title(zoo.DISPLAY[name]); ax.set_xlabel("epoch"); ax.set_ylabel(unit)
        if name in ("ae", "sae"):
            ax.set_yscale("log")
        if name == "sae" and "train_active_frac" in h["history"][0]:
            ax2 = ax.twinx()
            ax2.plot(ep, [e["train_active_frac"] for e in h["history"]], color="#888", lw=1, ls=":")
            ax2.set_ylabel("fraksi unit h aktif", color="#666")
            ax2.spines["right"].set_visible(True)
        ax.legend(frameon=False, fontsize=8)

    for k, name in enumerate(("gan", "dcgan")):
        ax = fig.add_subplot(gs[1, k])
        h = _hist(name)
        if h is None:
            ax.set_title(f"{zoo.DISPLAY[name]} (no log)"); continue
        ipe = h.get("iters_per_epoch", 1)
        il = np.array(h.get("iter_log", []))
        if len(il):
            ax.plot(il[:, 0] / ipe, il[:, 1], color="#2f6690", lw=0.4, alpha=0.35)
            ax.plot(il[:, 0] / ipe, il[:, 2], color="#b23a48", lw=0.4, alpha=0.35)
        ep = [e["epoch"] for e in h["history"]]
        ax.plot(ep, [e["loss_D"] for e in h["history"]], color="#2f6690", lw=1.8, label="Discriminator")
        ax.plot(ep, [e["loss_G"] for e in h["history"]], color="#b23a48", lw=1.8, label="Generator")
        ax.set_title(f"{zoo.DISPLAY[name]} (garis tipis = per 20 iterasi)")
        ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
        ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    h = _hist("ddpm")
    if h is not None:
        ep = [e["epoch"] for e in h["history"]]
        ax.plot(ep, [e["train_loss"] for e in h["history"]], color=PALETTE["ddpm"], lw=1.6, label="train")
        ax.plot(ep, [e["val_loss_ema"] for e in h["history"]], color=PALETTE["ddpm"], lw=1.6, ls="--",
                label="val (EMA)")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("DDPM: loss prediksi derau"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE(eps, eps_theta)")

    ax = fig.add_subplot(gs[2, :])
    model = zoo.load("ddpm", device, strict=False)
    if model is not None:
        fr = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0]
        steps = sorted({round(f * model.T) for f in fr}, reverse=True)
        g = torch.Generator(device=device).manual_seed(C.SEED)
        _, traj = model.sample(4, device, generator=g, keep_steps=set(steps))
        cols = torch.stack([traj[t] for t in steps], 1).flatten(0, 1)        # (4*10, 1, 28, 28)
        ax.imshow(grid_img(cols, len(steps)), cmap="gray", vmin=0, vmax=1)
        w = 30
        ax.set_xticks([w * i + 15 for i in range(len(steps))])
        ax.set_xticklabels([f"t={t}" for t in steps], fontsize=8)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax.set_title("Trajektori reverse diffusion DDPM: x_T -> x_0 (4 sampel)")
    fig.suptitle("Figur 5. Kurva loss training dan trajektori denoising DDPM", fontsize=12, y=0.985)
    save(fig, "fig5_loss_curves_ddpm_trajectory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    device = get_device()
    fns = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5}
    for i in args.only:
        set_seed(C.SEED)
        fns[i](device)


if __name__ == "__main__":
    main()
