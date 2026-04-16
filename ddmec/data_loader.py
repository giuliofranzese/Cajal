import os
import urllib.request
import pandas as pd


# ---------------------------------------------------------------------------
# Download URLs — fill these in before use
# ---------------------------------------------------------------------------
DOWNLOAD_URLS = {
    "atac": "https://huggingface.co/buckets/mustabou/mybucket/resolve/ATAC_LSI.csv?download=true",  # e.g. "https://example.com/ATAC_LSI.csv"
    "rna":  "https://huggingface.co/buckets/mustabou/mybucket/resolve/RNA_PCA.csv?download=true",  # e.g. "https://example.com/RNA_PCA.csv"
    "ann":  "https://huggingface.co/buckets/mustabou/mybucket/resolve/annotations.csv?download=true",  # e.g. "https://example.com/annotations.csv"
}


def _download_file(url: str, dest_path: str) -> None:
    """Download *url* to *dest_path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)


def load_pbmc_data(use_download=True, cache_dir=None, local_root=None):
    """Load PBMC Multiome CSV files.

    Tries direct URL download when `use_download` is True and all DOWNLOAD_URLS
    are set.  Downloaded files are cached under `cache_dir` (defaults to
    ``<repo_root>/.cache/pbmc``) so subsequent calls skip re-downloading.

    Falls back to local files under `local_root` (defaults to repo root).

    Returns: (source_tech, target_tech, meta_source, meta_target) as pandas.DataFrame
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if local_root is None:
        local_root = repo_root
    if cache_dir is None:
        cache_dir = os.path.join(repo_root, ".cache", "pbmc")

    # Relative paths inside the repo / cache
    files = {
        "atac": os.path.join("scTopoGAN", "Data", "PBMC Multiome", "ATAC_LSI.csv"),
        "rna":  os.path.join("scTopoGAN", "Data", "PBMC Multiome", "RNA_PCA.csv"),
        "ann":  os.path.join("scTopoGAN", "Data", "PBMC Multiome", "annotations.csv"),
    }

    # Try URL download if requested and all URLs are populated
    if use_download and all(DOWNLOAD_URLS.values()):
        try:
            paths = {}
            for name, rel in files.items():
                dest = os.path.join(cache_dir, rel)
                if not os.path.exists(dest):
                    print(f"Downloading {name} from {DOWNLOAD_URLS[name]} ...")
                    _download_file(DOWNLOAD_URLS[name], dest)
                paths[name] = dest

            source_tech = pd.read_csv(paths["atac"], header=0, index_col=0)
            target_tech = pd.read_csv(paths["rna"],  header=0, index_col=0)
            meta_source = pd.read_csv(paths["ann"],  header=0, index_col=0)
            meta_target = pd.read_csv(paths["ann"],  header=0, index_col=0)
            return source_tech, target_tech, meta_source, meta_target
        except Exception as e:
            print(f"URL download failed: {e}. Falling back to local files.")

    # Fallback to local files (repo layout)
    atac_path = os.path.join(local_root, files["atac"])
    rna_path  = os.path.join(local_root, files["rna"])
    ann_path  = os.path.join(local_root, files["ann"])

    if not (os.path.exists(atac_path) and os.path.exists(rna_path) and os.path.exists(ann_path)):
        raise FileNotFoundError(
            "One or more PBMC data files not found. Checked:\n"
            f"  {atac_path}\n  {rna_path}\n  {ann_path}\n"
            "Set `local_root` to your repo root or populate DOWNLOAD_URLS for direct download."
        )

    source_tech = pd.read_csv(atac_path, header=0, index_col=0)
    target_tech = pd.read_csv(rna_path,  header=0, index_col=0)
    meta_source = pd.read_csv(ann_path,  header=0, index_col=0)
    meta_target = pd.read_csv(ann_path,  header=0, index_col=0)

    return source_tech, target_tech, meta_source, meta_target






