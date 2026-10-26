"""Training entry point.

    python -m src.train --model mlp      # train first: also used to measure GAN mode coverage
    python -m src.train --model ae | sae | vae | gan | dcgan | ddpm
    python -m src.train --model all

Every run calls set_seed(42) and uses the same split from src/data.py.
Checkpoints -> checkpoints/<model>.pt, logs -> results/logs/<model>_history.json
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from . import config as C
from . import zoo
from .data import get_loaders
from .metrics import mode_coverage
from .models import EMA, build_ddpm, build_gan, gan_sample
from .utils import count_params, get_device, load_ckpt, save_ckpt, save_json, set_seed, to_sym

ORDER = ["mlp", "ae", "sae", "vae", "gan", "dcgan", "ddpm"]


def _log_path(name):
    return os.path.join(C.LOG_DIR, f"{name}_history.json")


def _summary(name, history, model, device, extra=None):
    times = [h["epoch_time_s"] for h in history]
    out = {"model": name, "params": count_params(model), "device": str(device),
           "epochs": len(history), "avg_epoch_time_s": sum(times) / len(times),
           "total_time_s": sum(times), "history": history}
    out.update(extra or {})
    save_json(out, _log_path(name))
    return out


# ------------------------------------------------------------------ MLP classifier
def train_mlp(device, epochs):
    train, val, test = get_loaders()
    model = zoo.build("mlp").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR["mlp"])
    hist, best = [], -1.0

    def accuracy(loader):
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in loader:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
                n += len(y)
        return correct / n

    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        dt = time.perf_counter() - t0
        va = accuracy(val)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": va, "epoch_time_s": dt})
        print(f"[mlp] ep {ep:3d} loss {tot / n:.4f} val_acc {va:.4f} ({dt:.1f}s)")
        if va > best:
            best = va
            save_ckpt("mlp", model=model.state_dict(), epoch=ep, val_acc=va)

    model.load_state_dict(load_ckpt("mlp", device)["model"])
    te = accuracy(test)
    ck = load_ckpt("mlp", device)
    save_ckpt("mlp", **ck, test_acc=te)
    flag = "OK" if te >= C.MLP_TARGET_ACC else f"BELOW TARGET {C.MLP_TARGET_ACC:.0%}"
    print(f"[mlp] best val_acc {best:.4f} | test_acc {te:.4f} [{flag}]")
    return _summary("mlp", hist, model, device, {"best_val_acc": best, "test_acc": te})


# ------------------------------------------------------------------ AE / Sparse AE / VAE
def train_autoencoder(name, device, epochs):
    train, val, _ = get_loaders()
    model = zoo.build(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR[name])
    hist, best = [], float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        t0, agg, n = time.perf_counter(), {}, 0
        for x, _ in train:
            x = x.to(device)
            loss, logs = model.compute_loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            logs["loss"] = loss.item()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v * len(x)
            n += len(x)
        dt = time.perf_counter() - t0

        model.eval()
        vloss = vmse = vn = 0
        with torch.no_grad():
            for x, _ in val:
                x = x.to(device)
                torch.manual_seed(C.SEED)            # fixed reparam noise for the VAE val loss
                l, _ = model.compute_loss(x)
                vloss += l.item() * len(x)
                vmse += F.mse_loss(model.reconstruct(x), x).item() * len(x)
                vn += len(x)
        rec = {"epoch": ep, **{f"train_{k}": v / n for k, v in agg.items()},
               "val_loss": vloss / vn, "val_mse": vmse / vn, "epoch_time_s": dt}
        hist.append(rec)
        extra = f" active {rec['train_active_frac']:.3f}" if "train_active_frac" in rec else ""
        print(f"[{name}] ep {ep:3d} train {rec['train_loss']:.4f} val {rec['val_loss']:.4f} "
              f"val_mse {rec['val_mse']:.5f}{extra} ({dt:.1f}s)")
        if rec["val_loss"] < best:
            best = rec["val_loss"]
            save_ckpt(name, model=model.state_dict(), epoch=ep, val_loss=best)
    return _summary(name, hist, model, device, {"best_val_loss": best})


# ------------------------------------------------------------------ Vanilla GAN / DCGAN
def _grad_norm(module):
    g = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
    return torch.cat(g).norm().item() if g else 0.0


def train_gan(name, device, epochs):
    train, _, _ = get_loaders()
    G, D = (m.to(device) for m in build_gan(name))
    optG = torch.optim.Adam(G.parameters(), lr=C.LR[name], betas=C.GAN_BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=C.LR[name], betas=C.GAN_BETAS)
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(64, G.z_dim, generator=torch.Generator().manual_seed(C.SEED)).to(device)
    clf = zoo.load("mlp", device, strict=False)
    if clf is None:
        print(f"[{name}] no MLP checkpoint -> mode coverage tracking disabled (train mlp first)")
    hist, iter_log, it = [], [], 0
    for ep in range(1, epochs + 1):
        G.train(); D.train()
        t0 = time.perf_counter()
        s = {"loss_D": 0.0, "loss_G": 0.0, "D_x": 0.0, "D_G_z": 0.0, "grad_G": 0.0, "grad_D": 0.0}
        nb = 0
        for x, _ in train:
            real = to_sym(x.to(device))
            b = real.shape[0]
            ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
            fake = G(torch.randn(b, G.z_dim, device=device))

            # D: max log D(x) + log(1 - D(G(z)))
            d_real, d_fake = D(real), D(fake.detach())
            lossD = bce(d_real, ones) + bce(d_fake, zeros)
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            s["grad_D"] += _grad_norm(D)
            optD.step()

            # G: non-saturating, max log D(G(z))
            d_gen = D(fake)
            lossG = bce(d_gen, ones)
            optG.zero_grad(set_to_none=True)
            lossG.backward()
            s["grad_G"] += _grad_norm(G)
            optG.step()

            s["loss_D"] += lossD.item(); s["loss_G"] += lossG.item()
            s["D_x"] += torch.sigmoid(d_real).mean().item()
            s["D_G_z"] += torch.sigmoid(d_fake).mean().item()
            nb += 1
            it += 1
            if it % 20 == 0:
                iter_log.append([it, lossD.item(), lossG.item()])
        dt = time.perf_counter() - t0
        rec = {"epoch": ep, **{k: v / nb for k, v in s.items()}, "epoch_time_s": dt}

        # Q3 diagnostic: gradient reaching G under the original saturating loss log(1 - D(G(z)))
        G.zero_grad(set_to_none=True)
        z = torch.randn(256, G.z_dim, device=device)
        sat = -bce(D(G(z)), torch.zeros(256, device=device))   # = mean log(1 - D(G(z)))
        sat.backward()
        rec["grad_G_saturating"] = _grad_norm(G)
        G.zero_grad(set_to_none=True)

        if ep % C.SNAPSHOT_EVERY == 0 or ep == epochs or ep == 1:
            G.eval()
            save_image(gan_sample(G, 64, device, z=fixed_z),
                       os.path.join(C.SAMPLE_DIR, f"{name}_epoch{ep:03d}.png"), nrow=8)
            if clf is not None:
                with torch.no_grad():
                    imgs = torch.cat([gan_sample(G, 500, device) for _ in range(4)])
                mc = mode_coverage(imgs, clf, device)
                rec["mode_entropy_ratio"] = mc["entropy_ratio"]
                rec["classes_covered"] = mc["classes_covered"]
        hist.append(rec)
        mc_txt = f" modeH {rec['mode_entropy_ratio']:.3f}" if "mode_entropy_ratio" in rec else ""
        print(f"[{name}] ep {ep:3d} D {rec['loss_D']:.3f} G {rec['loss_G']:.3f} "
              f"D(x) {rec['D_x']:.3f} D(G(z)) {rec['D_G_z']:.3f} |gG| {rec['grad_G']:.3f}{mc_txt} ({dt:.1f}s)")
    save_ckpt(name, G=G.state_dict(), D=D.state_dict(), epoch=epochs)
    return _summary(name, hist, G, device, {"params_D": count_params(D), "iters_per_epoch": nb,
                                            "iter_log": iter_log})


# ------------------------------------------------------------------ DDPM
def train_ddpm(device, epochs):
    train, val, _ = get_loaders()
    diff = build_ddpm().to(device)
    ema = EMA(diff)
    opt = torch.optim.Adam(diff.parameters(), lr=C.LR["ddpm"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    for ep in range(1, epochs + 1):
        diff.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, _ in train:
            x0 = to_sym(x.to(device))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = diff.compute_loss(x0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(diff)
            tot += loss.item() * len(x)
            n += len(x)
        dt = time.perf_counter() - t0

        vtot = vn = 0
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
            torch.manual_seed(C.SEED)
            for x, _ in val:
                l = ema.shadow.compute_loss(to_sym(x.to(device)))
                vtot += l.item() * len(x)
                vn += len(x)
        rec = {"epoch": ep, "train_loss": tot / n, "val_loss_ema": vtot / vn, "epoch_time_s": dt}
        hist.append(rec)
        print(f"[ddpm] ep {ep:3d} train {rec['train_loss']:.4f} val(ema) {rec['val_loss_ema']:.4f} ({dt:.1f}s)")
        save_ckpt("ddpm", model=diff.state_dict(), ema=ema.shadow.state_dict(), epoch=ep)
        if device.type == "cuda" and (ep % C.SNAPSHOT_EVERY == 0 or ep == epochs):
            g = torch.Generator(device=device).manual_seed(C.SEED)
            save_image(ema.shadow.sample(64, device, generator=g),
                       os.path.join(C.SAMPLE_DIR, f"ddpm_epoch{ep:03d}.png"), nrow=8)
    return _summary("ddpm", hist, diff.model, device, {"T": diff.T})


# ------------------------------------------------------------------ CLI
def run(name, device, epochs=None):
    set_seed(C.SEED)
    epochs = epochs or C.EPOCHS[name]
    print(f"=== {zoo.DISPLAY[name]} | {epochs} epochs | device {device} ===")
    if name == "mlp":
        return train_mlp(device, epochs)
    if name in ("ae", "sae", "vae"):
        return train_autoencoder(name, device, epochs)
    if name in ("gan", "dcgan"):
        return train_gan(name, device, epochs)
    if name == "ddpm":
        return train_ddpm(device, epochs)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ORDER + ["all"])
    ap.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    args = ap.parse_args()
    device = get_device()
    for name in (ORDER if args.model == "all" else [args.model]):
        run(name, device, args.epochs)


if __name__ == "__main__":
    main()
