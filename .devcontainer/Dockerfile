FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    ipykernel \
    scikit-learn \
    pandas \
    torchdiffeq

WORKDIR /workspace
