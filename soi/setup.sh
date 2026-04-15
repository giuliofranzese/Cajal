#!/usr/bin/env bash
# SOI Quickstart setup script
# Installs Python dependencies and clones the SOI repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/soi_repo"

echo "=== Installing Python dependencies ==="
pip install -q -r "${SCRIPT_DIR}/requirements.txt"


echo "=== Setup complete ==="
echo "You can now run the notebook: soi/notebook.ipynb"
