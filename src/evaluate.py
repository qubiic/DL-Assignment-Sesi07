"""Bagian IV - quantitative comparison of the six generative models on the test split.

    python -m src.evaluate                 # everything (GPU + CPU latency, FID)
    python -m src.evaluate --no-fid        # skip FID
    python -m src.evaluate --no-cpu        # skip CPU latency (DDPM on CPU takes a few minutes)

Outputs results/metrics.json, results/metrics_table.csv, results/metrics_table.md
"""
import argparse
import csv
import os

import numpy as np
import torch

from . import config as C
from . import zoo
from .data import get_loaders, get_splits
from .metrics import FIDScorer, laplacian_variance, measure_latency, mode_coverage, reconstruction_metrics
from .utils import count_params, get_device, load_json, save_json, set_seed

MODELS = ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]
LATENT = {"ae": "64", "sae": "64 (bottleneck) / 256 (sparse)", "vae": "64",
          "gan": "100 (prior noise)", "dcgan": "100 (prior noise)", "ddpm": "1x28x28 (dimensi citra)"}
RECON = ("ae", "sae", "vae")
GENERATIVE = ("vae", "gan", "dcgan", "ddpm")


def latency_fn(name, model, device, x64):
    """What 'one inference' means per model (Bagian IV: batch of 64)."""
    if name in ("ae", "sae"):
        return lambda: model.reconstruct(x64)                          # encode + decode
    if name == "vae":
        return lambda: model.sample(C.LATENCY_BATCH, device)            # prior -> decoder
    if name in ("gan", "dcgan"):
        z = torch.randn(C.LATENCY_BATCH, model.z_dim, device=device)
        return lambda: model(z)                                          # single generator pass
    if name == "ddpm":
        return lambda: model.sample(C.LATENCY_BATCH, device)            # T = 1000 reverse steps
    raise ValueError(name)


def stability(name, hist):
    h = hist["history"]
    if name in RECON or name == "ddpm":
        key = "val_loss_ema" if name == "ddpm" else "val_loss"
        v = [e[key] for e in h]
        tail = v[-5:] if len(v) >= 5 else v
        rel = (max(tail) - min(tail)) / abs(np.mean(tail)) if len(tail) > 1 else 0.0
        mono = sum(v[i + 1] > v[i] for i in range(len(v) - 1)) / max(1, len(v) - 1)
        label = "Stabil / Konvergen" if rel < 0.02 else "Masih turun / berfluktuasi"
        return {"label": label, "val_loss_final": v[-1], "rel_range_last5": rel,
                "frac_epochs_val_increase": mono}
    lg = np.array([e["loss_G"] for e in h])
    ld = np.array([e["loss_D"] for e in h])
    q = max(1, len(lg) // 4)
    cv_g = float(lg[-q:].std() / max(1e-8, abs(lg[-q:].mean())))
    ent = [e["mode_entropy_ratio"] for e in h if "mode_entropy_ratio" in e]
    out = {"cv_loss_G_last25pct": cv_g, "loss_G_final": float(lg[-1]), "loss_D_final": float(ld[-1]),
           "D_x_final": h[-1]["D_x"], "D_G_z_final": h[-1]["D_G_z"],
           "grad_G_nonsat_final": h[-1]["grad_G"], "grad_G_saturating_final": h[-1]["grad_G_saturating"],
           "mode_entropy_min": min(ent) if ent else None, "mode_entropy_final": ent[-1] if ent else None}
    collapse = ent and min(ent) < 0.85
    out["label"] = ("Osilasi tinggi" if cv_g > 0.15 else "Osilasi sedang" if cv_g > 0.05 else "Relatif stabil") \
        + (" / indikasi mode collapse" if collapse else "")
    return out


@torch.no_grad()
def generate(sample_fn, n, batch=500):
    out, done = [], 0
    while done < n:
        k = min(batch, n - done)
        out.append(sample_fn(k).cpu())
        done += k
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--fid-n", type=int, default=C.FID_N)
    args = ap.parse_args()
    set_seed(C.SEED)
    device = get_device()
    _, _, test_loader = get_loaders()
    splits = get_splits()
    x_test = splits["test"][0]
    x64 = x_test[:C.LATENCY_BATCH]
    clf = zoo.load("mlp", device, strict=False)

    fid = None
    if not args.no_fid:
        try:
            fid = FIDScorer(x_test, device)
        except Exception as e:
            print(f"[fid] disabled: {e}")

    prev = load_json(os.path.join(C.RESULTS_DIR, "metrics.json")) or {}
    out = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "cpu_threads": torch.get_num_threads(), "fid_n_generated": args.fid_n,
           "fid_n_real": len(x_test), "models": prev.get("models", {})}

    if fid is not None:
        n_ref = min(args.fid_n, len(splits["val"][0]))
        out["fid_reference_val_vs_test"] = fid.score_images(splits["val"][0][:n_ref])
        print(f"[fid] real-vs-real floor (val vs test): {out['fid_reference_val_vs_test']:.2f}")

    for name in args.models:
        model = zoo.load(name, device, strict=False)
        if model is None:
            print(f"[{name}] checkpoint missing - skipped")
            continue
        torch.manual_seed(C.SEED)
        hist = load_json(os.path.join(C.LOG_DIR, f"{name}_history.json")) or {}
        row = {"display": zoo.DISPLAY[name], "latent_dim": LATENT[name],
               "params": count_params(model.model if name == "ddpm" else model),
               "epochs": hist.get("epochs"), "train_time_per_epoch_s": hist.get("avg_epoch_time_s")}

        # reconstruction fidelity (Q1)
        if name in RECON:
            row.update(reconstruction_metrics(model.reconstruct, test_loader, device))
            with torch.no_grad():
                rec = torch.cat([model.reconstruct(x.to(device)).cpu() for x, _ in test_loader])
            row["sharpness_recon"] = laplacian_variance(rec)

        # latency (Q3, Q6)
        is_ddpm = name == "ddpm"
        reps, warm = (3, 1) if is_ddpm else (C.LATENCY_REPS, C.LATENCY_WARMUP)
        row["latency_ms"] = measure_latency(latency_fn(name, model, device, x64.to(device)),
                                            device, reps=reps, warmup=warm)
        row["latency_ms"]["device"] = str(device)
        if not args.no_cpu and device.type == "cuda":
            cpu = torch.device("cpu")
            mc = model.to(cpu)
            row["latency_cpu_ms"] = measure_latency(latency_fn(name, mc, cpu, x64), cpu,
                                                    reps=1 if is_ddpm else 20, warmup=0 if is_ddpm else 3)
            model = mc.to(device)
        print(f"[{name}] latency {row['latency_ms']['mean_ms']:.2f} ms / 64 samples")

        # generative quality, sharpness, mode coverage (Q1-Q3)
        if name in GENERATIVE:
            n = args.fid_n
            samples = generate(zoo.sampler(name, model, device), n)
            row["sharpness_samples"] = laplacian_variance(samples)
            if clf is not None:
                row["mode_coverage"] = mode_coverage(samples, clf, device)
            if fid is not None:
                row["fid"] = fid.score_images(samples)
                print(f"[{name}] FID {row['fid']:.2f}")

        if hist:
            row["stability"] = stability(name, hist)
        out["models"][name] = row
        save_json(out, os.path.join(C.RESULTS_DIR, "metrics.json"))     # checkpoint progress

    out["sharpness_real_test"] = laplacian_variance(x_test)
    save_json(out, os.path.join(C.RESULTS_DIR, "metrics.json"))
    write_tables(out)


def _fmt(v, nd=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_tables(out):
    header = ["Arsitektur Model", "Dim Laten", "Training Time / Epoch (s)", "Latency GPU (ms / batch 64)",
              "Latency CPU (ms / batch 64)", "Test MSE", "Test PSNR (dB)", "Test SSIM", "FID Score",
              "Kestabilan Training"]
    rows = []
    for name in MODELS:
        r = out["models"].get(name)
        if r is None:
            continue
        lat = r.get("latency_ms", {})
        latc = r.get("latency_cpu_ms", {})
        rows.append([
            r["display"], r["latent_dim"], _fmt(r.get("train_time_per_epoch_s"), 2),
            f"{lat['mean_ms']:.2f} ± {lat['std_ms']:.2f}" if lat else "N/A",
            f"{latc['mean_ms']:.2f}" if latc else "N/A",
            _fmt(r.get("test_mse"), 5) if name in RECON else "N/A (tanpa encoder)",
            _fmt(r.get("test_psnr"), 2) if name in RECON else "N/A",
            _fmt(r.get("test_ssim"), 4) if name in RECON else "N/A",
            _fmt(r.get("fid"), 2) if name in GENERATIVE else "N/A (latent voids)",
            r.get("stability", {}).get("label", ""),
        ])
    with open(os.path.join(C.RESULTS_DIR, "metrics_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if "fid_reference_val_vs_test" in out:
        md.append(f"\nFID referensi data riil (val vs test, n={out['fid_n_generated']}): "
                  f"{out['fid_reference_val_vs_test']:.2f}")
    with open(os.path.join(C.RESULTS_DIR, "metrics_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
