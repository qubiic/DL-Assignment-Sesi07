# Studi Komparatif Model Generatif pada Fashion-MNIST

Tugas Take-Home Deep Learning (D6421) Sesi 07: Generative Models.

| | |
|---|---|
| Nama | Ariya Saddhana Lius |
| NIM | 2802394456 |
| Kelas | _(isi kelas)_ |
| Dataset | Fashion-MNIST (70.000 citra 28x28 grayscale, 10 kelas) |

Repositori ini mengimplementasikan, melatih, dan mengevaluasi enam model generatif: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM. Selain itu ada eksperimen pembuktian bahwa bobot dan fitur laten MLP Classifier tidak dapat dipakai untuk image generation (Bagian 2 panduan).

## Struktur

```
src/
  config.py          seed, split, hyperparameter, path output
  data.py            loading Fashion-MNIST + split 80/10/10 (seed 42)
  models/
    autoencoder.py   Vanilla AE, Sparse AE, Decoder 64-256-784
    vae.py           VAE (ELBO: BCE + KL)
    gan.py           Vanilla GAN (MLP) dan DCGAN (konvolusional penuh)
    ddpm.py          U-Net, Gaussian diffusion (T = 1000), EMA
    mlp_classifier.py  MLP 784-256-64-10 + proyeksi transpos W^T
  train.py           training semua model
  mlp_experiment.py  Bagian 2, Tahap 2-5
  evaluate.py        tabel metrik Bagian IV (MSE, PSNR, SSIM, FID, latency)
  figures.py         Figur 1-5 Bagian V
  metrics.py, zoo.py, utils.py
notebooks/run_all_colab.ipynb   menjalankan seluruh pipeline di Google Colab
scripts/run_all.sh               pipeline lengkap untuk mesin lokal ber-GPU
results/                         log, metrik, dan figur (di-commit)
checkpoints/                     bobot model (tidak di-commit)
```

## Cara menjalankan

Paling mudah lewat Colab: buka `notebooks/run_all_colab.ipynb`, pilih runtime GPU (T4), lalu jalankan semua sel.

Di mesin lokal:

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

Atau per tahap:

```bash
python -m src.train --model mlp      # wajib pertama: dipakai juga untuk mengukur mode coverage GAN
python -m src.train --model ae       # lalu sae, vae, gan, dcgan, ddpm
python -m src.mlp_experiment         # Bagian 2 (butuh checkpoint mlp dan ae)
python -m src.evaluate               # tabel metrik
python -m src.figures                # lima figur
```

Uji cepat tanpa GPU (subset kecil, 1 epoch, DDPM dengan T = 50): `DL07_QUICK=1 bash scripts/run_all.sh`. Hasil mode ini hanya untuk memastikan kode berjalan, bukan untuk laporan.

## Protokol evaluasi terstandarisasi

- Seed 42 untuk `random`, `numpy`, `torch` (`set_seed` dipanggil di awal setiap script); cuDNN deterministic.
- Train (60k) dan test (10k) resmi digabung, lalu dibagi 80/10/10 dengan permutasi seed 42: 56.000 train, 7.000 val, 7.000 test. Semua model memakai indeks yang sama.
- `batch_size = 128`, `shuffle=True` hanya pada train loader.
- Checkpoint terbaik dipilih dengan val set; semua angka di tabel dihitung pada test set.
- FID: fitur Inception-v3 2048-d (torchmetrics). Statistik riil diambil dari 7.000 citra test, statistik sintetis dari 5.000 sampel per model. Sebagai acuan batas bawah, dihitung juga FID val-vs-test (data riil terhadap data riil).
- Latency: waktu menghasilkan satu batch 64 citra (GPU dan CPU), rata-rata 50 pengulangan setelah 10 warm-up. DDPM diukur dengan 1000 langkah reverse penuh.

## Ringkasan arsitektur

| Model | Arsitektur | Laten | Loss |
|---|---|---|---|
| MLP Classifier | 784-256-64-10, ReLU, dropout 0,2 | z_MLP 64 (post-ReLU) | Cross-entropy |
| Vanilla AE | Enc 784-256-64, Dec 64-256-784 (sigmoid) | 64 | MSE |
| Sparse AE | sama dengan AE | 64 + kode sparse 256 | MSE + λ·mean\|h\|, λ = 1e-3 |
| VAE | Enc 784-256-(μ, log σ²), Dec 64-256-784 | 64 | BCE + KL |
| Vanilla GAN | G: 100-256-512-1024-784 (tanh), D: 784-512-256-1 | 100 | BCE non-saturating |
| DCGAN | G: ConvT 100→256@7→128@14→1@28, D: Conv 1→64→128→1 | 100 | BCE non-saturating |
| DDPM | U-Net 64-128-128 + self-attention 7x7, EMA 0,999 | 1x28x28 | MSE prediksi derau |

Encoder AE sengaja dibuat identik dengan ekstraktor fitur MLP Classifier (784-256-64), dan decoder AE identik dengan decoder yang dilatih pada z_MLP beku (Tahap 4). Dengan begitu, perbedaan kualitas rekonstruksi antara keduanya hanya berasal dari objektif yang dipakai untuk melatih encoder.

## Output dan pemetaannya ke laporan

| File | Bagian laporan |
|---|---|
| `results/metrics_table.md`, `results/metrics.json` | IV (tabel komparatif), Q1, Q3, Q6 |
| `results/mlp_experiment.json` | III, Q5 |
| `results/interpolation_metrics.json` | Q4 |
| `results/logs/*_history.json` | Q3 (stabilitas training, gradien G, mode coverage per epoch) |
| `results/figures/fig1..fig5` | V |
| `results/samples/` | snapshot sampel GAN/DCGAN/DDPM per epoch |
