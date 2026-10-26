#!/usr/bin/env bash
# Full pipeline, in dependency order. Run from the repository root:  bash scripts/run_all.sh
set -euo pipefail

python -m src.data                          # download + print split sizes
python -m src.train --model mlp             # Tahap 1 (also the mode-coverage judge for GANs)
python -m src.train --model ae
python -m src.train --model sae
python -m src.train --model vae
python -m src.train --model gan
python -m src.train --model dcgan
python -m src.train --model ddpm
python -m src.mlp_experiment                # Bagian 2, Tahap 2-5
python -m src.evaluate                      # Bagian IV table
python -m src.figures                       # Bagian V, Figur 1-5
