# FKL — Functional KL Divergence Estimation

Estimate KL divergence between trajectory distributions using Functional Flow Matching (FFM) with the MINO-T architecture.

Contributors: WANG Chao, NEPOTE Luca, FRANZESE Giulio, MICHIARDI Pietro

## Setup

### Create the conda environment

```bash
conda env create -f requirement.yaml
conda activate fkl_muon
```

## Notebooks

Tutorial notebooks are in `code/notebooks/`. Each one runs a full experiment end-to-end: visualization, training, KL estimation, and result inspection.

| Notebook | Dataset | Config | Data dir | Grid (M) | Dims (D) |
|---|---|---|---|---|---|
| `tutorial_eb.ipynb` | Embryoid Body | `configs/eb.yaml` | `data/EB/` | 101 | 5 |
| `tutorial_hesc.ipynb` | human Embryonic Stem Cell | `configs/hesc.yaml` | `data/HESC/` | 121 | 5 |
| `tutorial_gm.ipynb` | Gaussian Mixture | `configs/gm.yaml` | `data/GM/` | 128 | 1 |

**EB and hESC** compare a reference method (`sbirr`) against five alternatives (`vsb`, `msbm`, `mfl`, `am`, `tigon`).
**GM** compares two distributions (A vs B).

### Running a notebook

1. Open a notebook in Jupyter or VS Code.
2. Set `GPU` in the configuration cell to your CUDA device index.
3. Run cells in order. Training logs and results are saved to `log/<dataset>/`.

## Project structure

```
FKL/
  configs/          # YAML experiment configs (eb, hesc, gm)
  data/             # Input trajectory data (.npy files)
  log/              # Training outputs and KL results (created at runtime)
  models/           # MINO-T and other model definitions
  notebooks/        # Tutorial notebooks
  scripts/          # Training and evaluation entry point (main.py)
  util/             # Utilities (plotting, checkpointing, etc.)
  functional_fm.py  # Functional flow matching implementation
  functional_kl.py  # KL divergence estimation
```
