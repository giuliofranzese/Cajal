# cajal — standalone copy for PBMC notebook

This folder contains a self-contained subset of the repo needed to run `pbmc.ipynb`.

Quick start
- create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- To load the PBMC files from Hugging Face set the env var `HF_REPO_ID` to the repo id (e.g. `user/dataset-repo`). If unset the notebook will fall back to the local files placed under `scTopoGAN/Data/PBMC Multiome/`.

- Run a quick import check:

```bash
python test_imports.py
```

Running the notebook
- Open `pbmc.ipynb` in JupyterLab/Notebook. Make sure the kernel uses the virtualenv above. Run the first cell to ensure the local `cajal/` package root is on `sys.path`.

Notes
- The notebook avoids hard-coded absolute paths. Use the `args.path` CLI option or `CKPT_PATH` environment variable to point to checkpoints when needed.
- If you want me to convert all internal imports to relative imports (e.g. `from .diffusion import ...`) I can do that next.
