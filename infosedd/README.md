README
======

Quick guide: installing `uv` and creating a uv environment with Python 3.12

1) Install uv (recommended):

```bash
# Official standalone installer (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pipx (recommended if you use pipx)
pipx install uv

# Or with pip (user install)
python -m pip install --user uv
```

2) Verify installation:

```bash
uv --version
```

3) Create a uv virtual environment using Python 3.12 (recommended):

```bash
# from your project directory (e.g. inside infosedd/)
cd /path/to/infosedd
# ensure Python 3.12 is available (uv will download if necessary):
uv python install 3.12
# create a project virtual environment using Python 3.12
uv venv --python 3.12
# install package and its dependencies from pyproject.toml (editable)
uv run python -m pip install -e .
```

Clone and install the mutinfo-diffusion repository (infosedd)
-----------------------------------------------------------

This repository (mutinfo-diffusion, project name `infosedd`) includes a pyproject.toml that declares the package and its dependencies. Install it as follows.

1) Clone the repository:

```bash
git clone https://github.com/AlbertoForesti/mutinfo-diffusion.git
cd mutinfo-diffusion
```

2) Recommended (uv-managed environment):

```bash
# ensure Python 3.12 is available (uv will download if needed)
uv python install 3.12
# create a project venv using Python 3.12
uv venv --python 3.12
```

3) Or using standard venv + pip (explicitly using Python 3.12):

```bash
# create a venv with Python 3.12
python3.12 -m venv .venv
# activate the venv (Linux/macOS)
source .venv/bin/activate
```

4) Install dependencies:

Install dependencies with pip:

```bash
# install package and required caduceus extras (editable)

cd mutinfo-diffusion

uv pip install -e .

# or with pip

pip install -e .
```

Note: These wheels are platform- and CUDA-specific. If installation fails, ensure you are on a matching Linux x86_64 host, have Python 3.12, and have compatible CUDA runtime libraries available. Installation downloads prebuilt wheels; runtime import may fail if system CUDA libraries are missing.

Notes
-----
- The project name is `infosedd` (importable as `import infosedd_synthetic` or `import infosedd_real_data`).
- pyproject.toml requires `Python >=3.10,<3.13` but `Python 3.12` is recommended for this tutorial for smooth install.