import sys
import os
import shutil
from pathlib import Path
print('Python', sys.version)
# Robustly locate soi_repo in parent directories or clone if missing
cwd = Path.cwd()
REPO_GIT_URL = 'https://github.com/MustaphaBounoua/soi.git'
soi_repo_dir = None
for p in [cwd] + list(cwd.parents)[:5]:
    cand = p / 'soi_repo'
    if cand.is_dir() and (cand / 'src').is_dir():
        soi_repo_dir = cand
        break
    cand2 = p / 'soi' / 'soi_repo'
    if cand2.is_dir() and (cand2 / 'src').is_dir():
        soi_repo_dir = cand2
        break
if soi_repo_dir is None:
    cand = cwd / 'soi_repo'
    if cand.is_dir() and (cand / 'src').is_dir():
        soi_repo_dir = cand
    else:
        cand2 = cwd / 'soi' / 'soi_repo'
        if cand2.is_dir() and (cand2 / 'src').is_dir():
            soi_repo_dir = cand2
if soi_repo_dir is None:
    print('soi_repo not found; cloning into', cwd / 'soi_repo')
    if shutil.which('git') is None:
        raise RuntimeError('git is required to clone soi_repo; please install git and re-run the notebook')
    import subprocess
    dest = cwd / 'soi_repo'
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['git', 'clone', '--depth', '1', REPO_GIT_URL, str(dest)])
    soi_repo_dir = dest
repo_root = str(soi_repo_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print('Added repo_root to sys.path', repo_root)
os.makedirs('results/soi_vbn', exist_ok=True)
vbn_pth = os.path.join(str(cwd), 'vbn.pth')
