#!/usr/bin/env bash
# SOI Quickstart setup script
# Sets up a virtual environment and installs Python dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/env"
REPO_DIR="${SCRIPT_DIR}/soi_repo"
REPO_GIT_URL="https://github.com/MustaphaBounoua/soi.git"

echo "=== Creating virtual environment ==="
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Virtual environment created at ${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

echo "=== Activating virtual environment ==="
source "${VENV_DIR}/bin/activate"

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -q -r "${SCRIPT_DIR}/requirements.txt"

echo "=== SOI repository check ==="
if [ -d "${REPO_DIR}" ]; then
    echo "SOI Repository already exists at ${REPO_DIR}"
    if [ -z "$(ls -A "${REPO_DIR}")" ]; then
        echo "Directory ${REPO_DIR} exists but is empty. Cloning ${REPO_GIT_URL}..."
        if ! command -v git >/dev/null 2>&1; then
            echo "git is required to clone the repository, but it's not installed. Please install git and re-run setup."
            exit 1
        fi
        git clone --depth 1 "${REPO_GIT_URL}" "${REPO_DIR}"
        echo "Cloned ${REPO_GIT_URL} into ${REPO_DIR}"
    else
        if [ -d "${REPO_DIR}/.git" ]; then
            echo "Existing git repository detected at ${REPO_DIR}. Attempting to pull latest changes..."
            if command -v git >/dev/null 2>&1; then
                git -C "${REPO_DIR}" pull --ff-only || echo "Warning: git pull failed; repository may have local changes."
            else
                echo "git not found; skipping pull."
            fi
        else
            echo "Directory ${REPO_DIR} is not empty; skipping clone."
        fi
    fi
else
    echo "SOI Repository not found at ${REPO_DIR}. Cloning ${REPO_GIT_URL}..."
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to clone the repository, but it's not installed. Please install git and re-run setup."
        exit 1
    fi
    git clone --depth 1 "${REPO_GIT_URL}" "${REPO_DIR}"
fi

echo "=== Setup complete ==="
echo "To use the environment, run: source soi/env/bin/activate"
echo "You can now run the notebook: soi/notebook_soi_vbn.ipynb"
