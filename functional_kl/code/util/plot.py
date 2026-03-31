import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection


# -----------------------
# Helper: keep only first 3 dims everywhere
# -----------------------
def first3(arr, name=""):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected (N,T,D), got {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError(f"{name}: need at least 3 dims, got {arr.shape[-1]}")
    return arr[..., :3]  # (N,T,3)



def add_colored_trajectories_3d(ax, trajs_NT3, idx, cmap, norm, alpha=0.55, lw=1.2):
    segs_xy, seg_z, seg_t = [], [], []

    for i in idx:
        xyz = np.asarray(trajs_NT3[i])
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            continue

        good = np.isfinite(xyz).all(axis=1)
        if good.sum() < 2:
            continue
        xyz = xyz[good]

        T = xyz.shape[0]
        t = np.linspace(0.0, 1.0, T)

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        xy0 = np.stack([x[:-1], y[:-1]], axis=1)
        xy1 = np.stack([x[1:],  y[1:]],  axis=1)
        segs = np.stack([xy0, xy1], axis=1)

        z_mid = 0.5 * (z[:-1] + z[1:])
        t_mid = 0.5 * (t[:-1] + t[1:])

        segs_xy.append(segs)
        seg_z.append(z_mid)
        seg_t.append(t_mid)

    if not segs_xy:
        return None

    segs_xy = np.concatenate(segs_xy, axis=0)
    seg_z   = np.concatenate(seg_z,   axis=0)
    seg_t   = np.concatenate(seg_t,   axis=0)

    lc = LineCollection(segs_xy, cmap=cmap, norm=norm)
    lc.set_array(seg_t)
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    ax.add_collection3d(lc, zs=seg_z, zdir="z")
    return lc

def plot_black_trajectories(ax, traj_arr, n_traj=250, alpha=0.35, lw=1.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    N = traj_arr.shape[0]
    n_traj = min(n_traj, N)
    if n_traj <= 0:
        return
    idx = rng.choice(N, size=n_traj, replace=False) if N > n_traj else np.arange(n_traj)
    for j in idx:
        ax.plot(traj_arr[j, :, 0], traj_arr[j, :, 1], traj_arr[j, :, 2],
                color="k", alpha=alpha, lw=lw)

def scatter_sbirr_points_at_taus_3d(ax, sbirr_NT3, taus, idx_traj,
                                     cmap, norm,
                                     pts_s=90, pts_edge_lw=1.4,
                                     halo_s=180, halo_alpha=0.95):
    """
    Plot SBIRR points at given taus for the SAME trajectories idx_traj
    (identical in every subplot).
    """

    _, T, _ = sbirr_NT3.shape
    idx_traj = np.asarray(idx_traj, dtype=int)

    for tt in taus:
        ti = int(np.clip(np.round(tt * (T - 1)), 0, T - 1))
        sel = sbirr_NT3[idx_traj, ti, :]  # (n_val,3)

        good = np.isfinite(sel).all(axis=1)
        sel = sel[good]
        if sel.shape[0] == 0:
            continue

        col = cmap(norm(float(tt)))

        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2],
                   s=halo_s, color="white", alpha=halo_alpha,
                   edgecolors="white", linewidths=0.0,
                   depthshade=False)

        # colored points with strong black edge
        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2],
                   s=pts_s, color=col, alpha=1.0,
                   edgecolors="k", linewidths=pts_edge_lw,
                   depthshade=False)
        
