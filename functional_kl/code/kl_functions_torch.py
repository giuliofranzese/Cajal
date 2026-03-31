import numpy as np
import scipy
from scipy.special import lambertw # for importance sampling
import torch
import math

# # functions

################################################################################
# functions
################################################################################
# ---- build fourier basis ----
 
def build_fourier_Phi(M, N,
                      device=None,
                      basis = 'fourier',
                      include_endpoint_cosine=True, 
            ): 

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if basis == 'fourier':

        x_array = np.arange(M)/M  # endpoint=False # [0, 1)
        ks      = np.arange(-N, N+1)

        
        Phi = np.exp(1j * 2*np.pi * np.outer(x_array, ks))

         
        Phi /= np.sqrt(M)
       

    elif basis == 'cosine':
        # sampling grid on [0, L]
        x_array = np.linspace(0.0, 1.0, int(M), endpoint=bool(include_endpoint_cosine))
        ks = np.arange(0, int(N) + 1)

        # raw cosine basis (M, N+1)
        # use x/L so that omega_n = pi*n/L
        Phi = np.cos(np.pi * np.outer(x_array, ks)) 

        # column-normalize under discrete ℓ2 
        col_norm = np.sqrt( np.sum(Phi ** 2, axis=0, keepdims=True) )
        col_norm = np.clip(col_norm, 1e-12, None)
        Phi = (Phi / col_norm) 



    x_array = torch.tensor(x_array,device=device, )
    ks = torch.tensor(ks,device=device, )
    Phi = torch.tensor(Phi,device=device, 
                       dtype=torch.complex64 if basis == 'fourier' else torch.float32)

    return x_array, ks, Phi
 
 
def mean_coeffs(m1k_freq, m1k_scale, 
                
                D, N, 
                
                M, 
                
                func_type='sin',
                device=None, 
                return_data = False,
                basis = 'fourier'):
    # ===== non-integer freq path (e.g., 0.5) =====
    freq_val = m1k_freq # 1.0 # 0.5

 
    # amplitude factor, matches numpy: amp = m1k_scale * sqrt(M)
    amp = torch.as_tensor(m1k_scale, device=device, )
    # amp = amp.reshape(1, -1).expand(1, D) # broadcast to all D



    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, ks, Phi = build_fourier_Phi(M, N, basis=basis)
    # print('compute Phi in mean coeffs', M, N, Phi.shape)

    # sample x in [0,1) with M points
    # x = torch.arange(M, device=device, ) / M
    phase = 2 * math.pi * freq_val * x  # (M,)

    if func_type == 'sin':
        base = torch.sin(phase)
    elif func_type == 'cos':
        base = torch.cos(phase)
    elif func_type == 'sin+cos':
        base = torch.sin(phase) + torch.cos(phase)
    else:
        raise ValueError("func_type must be 'sin', 'cos', or 'sin+cos'")

    # μ(x_m) = base(x_m) * A * sqrt(M)
    mu = base[None, :, None].repeat(1, 1, D) * amp  # (M, D), real
    # print(mu.shape, mu.min().item(), mu.max().item(), m1k_scale)
    # print(x.shape, mu.shape )
    # plt.plot(x.squeeze().detach().cpu().numpy(), mu.squeeze().detach().cpu().numpy())

    mu_hat = project_to_basis(mu, Phi)  # (2N+1, D)
   
    if return_data:
        return mu_hat, mu

    return mu_hat

# ---- Covariance: Matérn with periodic λ_k ---- 

def matern_lambda_periodic_from_ks(nu, ell, sigma2,
                                   ks, M,
                                   white = False, 
                                   device=None, 
                                   basis = 'fourier',):
 
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # lam^2 = 2 nu / ell^2
    lam2 = 2.0 * nu / (ell ** 2)

    # w^2 = (2πk)^2
    if  basis == 'fourier':
        w2 = (2.0 * np.pi * ks) ** 2
    elif basis == 'cosine':
        w2 = (np.pi * ks ) ** 2

    # cov_hat_unnorm = (a2 + w2)^(-(nu+1/2))
    shape = - (nu + 0.5)
    
    cov_hat_unnorm = (lam2 + w2).to(dtype=torch.float64) ** (shape)  # real, positive

   
    # target = (lam_pre ** (2.0 * nu_t)) * (math.pi ** 0.5) * sigma2_t * 2 * math.gamma(nu + 0.5) / math.gamma(nu)
    target = sigma2 * M
    cov_hat = target * cov_hat_unnorm / torch.sum( cov_hat_unnorm.real )
    cov_hat = torch.clamp_min(cov_hat.real, 0)

    cov_hat = cov_hat.to(dtype=torch.float32)
 
    if white:
        cov_hat = torch.ones_like(cov_hat) * sigma2
 
    return cov_hat

# ---- velocity , v_diff ----
def get_vk_vdiffk_array(m1, c, k, t, 
                        r = None,
                        return_item = 'vdiff' ,
                        device=None,   ):

    if device is None:
        device = m1.device if torch.is_tensor(m1) else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    m1 = torch.as_tensor(m1, device=device)
    c  = torch.as_tensor(c,  device=device)
    k  = torch.as_tensor(k,  device=device)
    t  = torch.as_tensor(t,  device=device)

    # c, k are (K,) 
    c = c.unsqueeze(0)
    k = k.unsqueeze(0) 
    # (1, K)

    

    # t
    if t.ndim == 0 or t.numel() == 1: # scalar
        t = t.view(1,1)
    elif return_item == 'v' and t.ndim == 1 and r is not None and t.shape[0] == r.shape[0]:
        t = t.unsqueeze(-1)
    else:
        raise ValueError(f"{t.shape} is wrong!")
    # (B, K)
 
    numerator   = t * c - (1.0 - t) * k
    denominator = (1.0 - t)**2 * k + t**2 * c

    g = numerator / denominator   
    # (B, K)

    if return_item == 'v':
        if r is None:
            raise ValueError("r must be provided when return_item == 'v'")
        r = torch.as_tensor(r, device=device)  # (B, K, D)
        B, K, D = r.shape

        g = g.unsqueeze(-1)
        t = t.unsqueeze(-1)

        for variable in [m1, g, r, t, m1]:
            assert variable.ndim == 3
   

        
        v_mean = m1 + g * (r - t * m1)
        return v_mean

    elif return_item == 'vdiff':
    
        v_diff = m1 + g.view(1, -1, 1) * (-t.view(1, -1, 1) * m1)
        return v_diff

    else:
        raise ValueError("return_item must be 'v' or 'vdiff'")


# ---- Integration over FFM's t ----
def riemann_sum(y, x, method="left",
                device=None, ):
    """
    Compute a 1D Riemann sum for samples y = f(x) on nodes x.
    Supports nonuniform spacing.

    method: 'left' | 'right' | 'midpoint' | 'trap'  (trap = trapezoid)
    """
    if device is None:
        # prefer y's device if it's already a tensor
        if torch.is_tensor(y):
            device = y.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = torch.as_tensor(y, device=device)
    x = torch.as_tensor(x, device=device)

    dx = x[1:] - x[:-1]  # (M-1,)

    if method == "left":
        # y[..., :-1] * dx broadcasts over leading dims
        return torch.sum(y[..., :-1] * dx, dim=-1)

    elif method == "right":
        return torch.sum(y[..., 1:] * dx, dim=-1)

    elif method == "midpoint":
        y_mid = 0.5 * (y[..., :-1] + y[..., 1:])
        return torch.sum(y_mid * dx, dim=-1)

    elif method in ("trap", "trapezoid"):
        y_mid = 0.5 * (y[..., :-1] + y[..., 1:])
        return torch.sum(y_mid * dx, dim=-1)

    else:
        raise ValueError("Unknown method: choose 'left' | 'right' | 'midpoint' | 'trap'")
  
 

def cumulative_riemann(y, x, method="left"):
    """
    Prefix Riemann sums:
        out[n] ≈ ∫_0^{x[n]} y(s) ds
    y: (..., M)
    x: (M,)
    returns: (..., M)
    """
    dx = x[1:] - x[:-1]  # (M-1,)

    if method == "left":
        inc = y[..., :-1] * dx        # (..., M-1)
    elif method == "right":
        inc = y[..., 1:] * dx
    elif method in ("midpoint", "trap", "trapezoid"):
        inc = 0.5 * (y[..., :-1] + y[..., 1:]) * dx
    else:
        raise ValueError("method must be left/right/midpoint/trap")

    # prefix sum with 0 at start
    out = torch.zeros_like(y)
    out[..., 1:] = torch.cumsum(inc, dim=-1)
    return out


def integration_t(kl_result, t_list, sum_way='riemann',
                        device=None):
    """
    Integrate KL values over t_list.

    Args
    ----
    kl_result : (n_samples, n_t, D) torch tensor or array-like
    t_list    : (n_t,) torch tensor or array-like
    sum_way   : 'uniform' | 'riemann' | 'trap' | 'simpson'
        - 'uniform'  : simple mean over t
        - 'riemann'  : left Riemann sum
        - 'trap'     : trapezoid rule (nonuniform OK)
        - 'simpson'  : composite Simpson (requires uniform spacing;
                        otherwise falls back to 'trap')

    Returns
    -------
    KL_est : (n_samples, D) torch tensor
    """
    if device is None:
        device = kl_result.device if torch.is_tensor(kl_result) else (
            t_list.device if torch.is_tensor(t_list) else
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    kl_result = torch.as_tensor(kl_result, device=device)
    t_list = torch.as_tensor(t_list, device=device)

   
    dtype = kl_result.dtype if kl_result.is_floating_point() else torch.float32
    kl_result = kl_result.to(dtype)
    t_list = t_list.to(dtype)

    # shapes
    # kl_result: (S, T, D)
    S, T, D = kl_result.shape
    assert t_list.shape == (T,)

    if sum_way == 'uniform':
        # mean over t dimension
        return kl_result.mean(dim=1)  # (S, D)

    dx = t_list[1:] - t_list[:-1]            # (T-1,)
    dx_b = dx.view(1, T-1, 1)                # broadcast to (1, T-1, 1)

    if sum_way == 'riemann':
        # left Riemann: sum y_t * dx_t
        y_left = kl_result[:, :-1, :]        # (S, T-1, D)
        return (y_left * dx_b).sum(dim=1)    # (S, D)

    elif sum_way == 'trap':
        # trapezoid: sum 0.5*(y_t + y_{t+1}) * dx_t
        y_mid = 0.5 * (kl_result[:, :-1, :] + kl_result[:, 1:, :])
        return (y_mid * dx_b).sum(dim=1)

    elif sum_way == 'simpson':
        # Composite Simpson for uniform spacing.
        # Needs odd number of points (even number of intervals).
        if T < 3 or (T % 2 == 0):
            # fall back to trap
            y_mid = 0.5 * (kl_result[:, :-1, :] + kl_result[:, 1:, :])
            return (y_mid * dx_b).sum(dim=1)

        # check uniformity (tolerant)
        h = dx.mean()
        if torch.max(torch.abs(dx - h)) > 1e-6 * torch.abs(h):
            # nonuniform spacing -> fall back to trap
            y_mid = 0.5 * (kl_result[:, :-1, :] + kl_result[:, 1:, :])
            return (y_mid * dx_b).sum(dim=1)

        # Simpson weights: 1,4,2,4,...,2,4,1
        w = torch.ones(T, device=device, )
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w_b = w.view(1, T, 1)  # (1, T, 1)

        return (h / 3.0) * (kl_result * w_b).sum(dim=1)

    else:
        raise ValueError("sum_way must be 'uniform' | 'riemann' | 'simpson' | 'trap'")



def sample_trunc_t_over_1mt(te, size, rng=None,
                                  device=None, 
                                  sort = False, return_Z=False):
    """
    Hybrid sampler:
      - Torch/GPU for sampling u, masks, branch2, bookkeeping
      - SciPy lambertw (CPU) for branch1 only

    Returns torch tensor on `device`.
    """
    te_t = torch.as_tensor(te, )
    if not (0.0 < float(te_t) < 1.0):
        raise ValueError("te must be in (0,1).")

    if device is None:
        device = te_t.device if te_t.is_cuda else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    te_t = te_t.to(device=device, )

    # RNG / generator
    # if rng is None:
    #     rng = torch.Generator(device=device)
    #     rng.manual_seed(int(3))

    # u on GPU
    u = torch.rand(size, device=device, generator=rng)

    # constants on GPU
    Z  = -torch.log1p(-te_t)          # -log(1-te)
    uc = (Z - te_t) / Z               # split point

    t = torch.empty_like(u)

    mask1 = (u <= uc)
    mask2 = ~mask1

    # ---- branch 1 (Lambert W on CPU) ----
    if mask1.any():
        # arg = -exp(-Z*u - 1)   in (-1/e, 0)
        arg_gpu = -torch.exp(-Z * u[mask1] - 1.0)

        # move ONLY arg to CPU numpy float64 (SciPy wants numpy)
        arg_cpu = arg_gpu.detach().to("cpu", dtype=torch.float64).numpy()

        # SciPy principal branch W0, real-valued here
        w_cpu = lambertw(arg_cpu, k=0).real  # numpy float64

        # back to GPU
        w_gpu = torch.from_numpy(w_cpu).to(device=device, dtype = t.dtype )

        t[mask1] = 1.0 + w_gpu

    # ---- branch 2 (pure GPU) ----
    if mask2.any():
        # t = 1 + ((1-te)/te) * (log(1-te) + Z*u)
        # log(1-te) = -Z
        t[mask2] = 1.0 + ((1.0 - te_t) / te_t) * (-Z + Z * u[mask2])

    t = t.clamp(0.0, 1.0)

    if sort:
        orig_shape = t.shape
        t_sorted, _ = torch.sort(t.reshape(-1))  # GPU sort
        t = t_sorted.reshape(orig_shape)

    if return_Z:
        return t, Z
    else:
        return t


# ---- KL comparison ----
def kl_gaussian_mean_highD_GT(m_hat, lam_k, Sigma_D=None):
    """
    KL( N(m_hat, K) || N(0, K) ) = 1/2 ∑_k m_k^T C_k^{-1} m_k,
    where C_k = λ_k * Σ_D.

    Parameters
    ----------
    m_hat : (K, D)
        mean coefficients in Fourier/eigen basis
    lam_k : (K,)
        spectral variances λ_k > 0
    Sigma_D : (D,D) or None
        cross-output covariance; if None, uses identity.
    eps : float
        stability term

    Returns
    -------
    KL : float
        scalar KL divergence
    """
    K, D = m_hat.shape

    if Sigma_D is None:
        # Independent outputs
        inv_lam = 1.0 / lam_k[:, None] 
        term = (torch.abs(m_hat)**2) * inv_lam
        KL = 0.5 * torch.sum(term)
        return KL.real    
    else:
        # General correlated outputs
        # Invert Sigma_D once
        Sigma_inv = torch.linalg.inv(Sigma_D)

        KL = 0.0
        for k in range(K):
            m_k = m_hat[k, :].reshape(D, 1)
            lam_inv = 1.0 / lam_k[k] 
            KL += (m_k.T @ (lam_inv * Sigma_inv) @ m_k).item()
        return 0.5 * KL.real


# ---------- 采样 Xi (显式传 N) ----------

def sample_xi_full_highD(N, D, 
                         Sigma_D=None, seed=None, 
                         n_samples=1,
                         device=None,
                         basis = 'fourier' ):
    """
    Sample Xi[k] ∈ C^D, k=-N..N, with:
      - Xi_0 real
      - Hermitian symmetry: Xi_-k = conj(Xi_k)

    Args
    ----
    N, D : int
    Sigma_D : (D, D) torch tensor or array-like, optional
        Desired covariance across D. If provided, we apply Xi @ L^T,
        where L = chol(Sigma_D + eps I).
    seed : int, optional
    n_samples : int
    device : torch.device or str, optional
    dtype : torch.float32 or torch.float64 (real dtype)

    Returns
    -------
    Xi : (n_samples, 2N+1, D) complex torch tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generator for reproducibility
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    # complex dtype matching real dtype
    cdtype = torch.complex64  



    if basis == 'cosine':
        Xi = torch.randn((n_samples, N + 1, D), device=device, generator=gen, dtype=torch.float32)

    elif basis == 'fourier':

        # allocate
        Xi = torch.empty((n_samples, 2 * N + 1, D), device=device, dtype=cdtype)
        c = N  # index for k=0

        # k = 0 (real)
        Xi0 = torch.randn((n_samples, D), device=device,  generator=gen)
        Xi[:, c, :] = Xi0.to(cdtype)

        # k > 0 : complex standard normal with variance 1 per complex dim
        Xi_pos  = torch.randn((n_samples, N, D, 2),
            device=device,  generator=gen)
        Xi_pos = torch.view_as_complex(Xi_pos) / math.sqrt(2.0)

        Xi[:, c+1:, :] = Xi_pos
        # k < 0: conjugate + reverse in k
        Xi[:, :c, :] = torch.conj(Xi_pos.flip(dims=(1,)))



    # apply desired D-covariance if provided
    if Sigma_D is not None:
        Sigma_D = torch.as_tensor(Sigma_D, device=device, )
        epsI = 1e-12 * torch.eye(D, device=device, )
        L = torch.linalg.cholesky(Sigma_D + epsI)  # (D, D), real lower-tri

     
        Xi = Xi @ L.T.to(cdtype)

    return Xi

 

# ---------- iFFT reconstruction ----------
def reconstruct_from_basis(Xhat, Phi):
    """
    Inverse / reconstruction:
        X = Φ Xhat
    Xhat: (n_samples, 2N+1, D) complex
    Phi : (M, 2N+1) complex

    Returns:
        X_data: (n_samples, M, D) real
    """
    # (1, M, 2N+1) @ (n_samples, 2N+1, D) -> (n_samples, M, D)
    X = Phi.unsqueeze(0) @ Xhat
    return X.real  # Hermitian symmetry => real-valued signal


# ---------- FFT projection ----------
def project_to_basis(X, Phi):
    """
    Forward transform (time -> frequency):
        Xhat = Φᴴ X

    X   : (n_samples, M, D) real/complex
    Phi : (M, 2N+1) complex (column-orthonormal)

    Returns:
        Xhat: (n_samples, 2N+1, D) complex
    """
    # (1, 2N+1, M) @ (n_samples, M, D) -> (n_samples, 2N+1, D)
    X = X.to(Phi.dtype) 
    Xhat = Phi.conj().T.unsqueeze(0) @ X
    return Xhat


# ---------- sample in basis + (optional) reconstruct ----------
def sample_gp_in_basis(lam_k,
                             N, D, Sigma_D=None, seed=None,
                             Phi=None, ifft=False, M=None,
                             n_samples=1,
                             device=None,
                             basis = 'fourier',):
    """
    Generate periodic Matérn GP samples (D-dim outputs) in Fourier basis.

    Args
    ----
    lam_k : (2N+1,) real, nonnegative spectrum on ks=-N..N
    N, D, M : int
    Sigma_D : (D,D) optional covariance across output dims
    seed : int optional
    Phi : (M, 2N+1) complex Fourier basis (Torch tensor)
    ifft : bool, if True also return X_data
    n_samples : int
    device, dtype : torch device / real dtype

    Returns
    -------
    if ifft:
        Xhat : (n_samples, 2N+1, D) complex
        X_data : (n_samples, M, D) real
    else:
        Xhat only
    """
    if device is None:
        if torch.is_tensor(lam_k):
            device = lam_k.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lam_k = torch.as_tensor(lam_k, device=device, )
    if basis == 'fourier':
        assert lam_k.shape == (2 * N + 1,)
    elif basis == 'cosine':
        assert lam_k.shape == (N + 1,)

    # sample Hermitian Xi on GPU
    Xi = sample_xi_full_highD(
        N, D, Sigma_D=Sigma_D, seed=seed,
        n_samples=n_samples, device=device, 
        basis = basis
    )
    if basis == 'fourier':
        assert Xi.shape == (n_samples, 2 * N + 1, D)
    elif basis == 'cosine':
        assert Xi.shape == (n_samples, N + 1, D)

    # spectral scaling: Xhat = sqrt(lam_k) * Xi
    # (1, 2N+1, 1) * (n_samples, 2N+1, D)
    Xhat = torch.sqrt(lam_k).view(1, -1, 1).to(Xi.dtype) * Xi

    if ifft:
        if Phi is None:
            raise ValueError("Phi must be provided when ifft=True")
        if M is None:
            M = Phi.shape[0]

        if basis == 'fourier':
            assert Phi.shape == (M, 2 * N + 1), print(Phi.shape, (M, 2 * N + 1))
        elif basis == 'cosine':
            assert Phi.shape == (M, N + 1), print(Phi.shape, (M, N + 1))

        X_data = reconstruct_from_basis(Xhat, Phi)
        return Xhat, X_data
    else:
        return Xhat








class sample_X_data(object):
    def __init__(self,  
                nu, ell,  sigma2,
                    M, upsample,
                    N, D,
                    white = False,
                    basis = 'fourier',
                    ):
        
        self.nu=nu
        self.ell=ell
        self.sigma2=sigma2
        self.white = white
        
        self.M = M
        self.N = N
        self.D = D

        self.basis = basis

        self.upsample = upsample
        self.pre_compute(upsample)
 

    def pre_compute(self, upsample=1):
        
        x_array, ks, Phi = build_fourier_Phi(int(self.M*upsample), self.N, 
                                             basis=self.basis,
                                             )

        if self.basis == 'fourier':
            assert Phi.shape == (int(self.M*upsample), 2*self.N+1)
        elif self.basis == 'cosine':
            assert Phi.shape == (int(self.M*upsample), self.N+1)


        self.Phi = Phi
    
        lam_k = matern_lambda_periodic_from_ks(nu=self.nu, ell=self.ell, sigma2=self.sigma2,
                                            ks=ks, M=int(self.M*upsample),
                                            white = self.white,
                                            basis=self.basis,
                                            )
        self.lam_k = lam_k

 

        print(f'pre_compute again for upsample={upsample}')
    
    def sample(self, n_samples, upsample=1, return_hat=False):
        if self.upsample != upsample:
            self.upsample = upsample
            self.pre_compute(upsample)
            

        X0_hat, X0_data = sample_gp_in_basis(self.lam_k, 
                        N=self.N, D=self.D, Sigma_D=None, seed=None, 
                        ifft = True, 

                        Phi = self.Phi, M=int(self.M*upsample), 
                        n_samples=n_samples, 

                        basis=self.basis, 
                        )
 
        if self.white:
            X0_data = torch.randn_like(X0_data)
        
        if not return_hat:
            return X0_data
        else:
            return X0_hat, X0_data






def get_noise_amp(x1_time_mean0, amp_way=None ):
    B, N, C = x1_time_mean0.shape 

    noise_amp = []
    for c in range(C): 
        # channel c
        x1 = x1_time_mean0[..., c]
        # print(x1.shape)


        # spatial_variance = (x1_time_mean0**2).mean(dim=(1,))
        # print(spatial_variance.mean())

        # Plot the sample
        # plot_samples_1d(x1_time_mean0, num_plots=5)

        # kvals, Abins_w1, spectrum_amplitudes, knrm = get_Fourier_spectrum_real(x1,  return_spectrum_amplitudes=True)
        

        # Use rFFT for real signals
        X = torch.fft.rfft(x1, dim=1, norm="ortho")             # (B, N//2+1) 
        spectrum_amplitudes = (X.abs() ** 2).mean(dim=0)        # mean over all the samples       # (N//2+1)
        assert len(spectrum_amplitudes.shape) == 1
        # print(spectrum_amplitudes.shape)

        

        new_spectrum = spectrum_amplitudes



        # integer frequency index in "cycles per box" units (same as your * npix)
        
        # npix = X.shape[-1]
        # kfreq = np.fft.rfftfreq(npix) * npix
        # knrm  = np.abs(kfreq)  
        
        knrm = torch.arange(1, spectrum_amplitudes.shape[0]+1, device =  spectrum_amplitudes.device).float() #  npix+1   torch.arange((n + 1) // 2) / (d * n)
        assert knrm.shape == spectrum_amplitudes.shape
        if amp_way == '*knrm':
            new_spectrum *=   knrm   #  * torch.tensor((       np.sqrt(knrm))).float()
        elif amp_way == '*knrm_1d5': # for fucking AM on Repre
            new_spectrum *=   (knrm.pow(1.5))
        elif amp_way == '*knrm_2': # for fucking AM on Repre
            new_spectrum *=   (knrm.pow(2))
        elif amp_way == '*knrm_0.5': # for Petal which TI are similar
            new_spectrum *=   (knrm.pow(0.5))
        elif amp_way == None: 
            new_spectrum = new_spectrum
        

        # Floor small eigenvalues to avoid 1/lam_k explosion in KL estimation
        eps_floor = 0.0 #new_spectrum.max() * 5e-4

        amp = torch.sqrt(new_spectrum.clamp_min(eps_floor))     # clamp to avoid explosion              # amplitude per bin
        
        noise_amp.append(amp)
        

  
 
    
    
    noise_amp = torch.stack(noise_amp, dim=1) # (N//2+1, C,)
    assert noise_amp.shape[-1] == C
    return noise_amp




# class noise_x0_class:
#     def __init__(self, 
#                  x1_time_mean0, 
#                  noise_amp=None, 
#                  white_noise = False, 
#                  amp_way  =None,
#                  ):

#         B, N, C = x1_time_mean0.shape 
#         self.N = N
#         self.C = C

#         self.device = "cuda:0" # x1_time_mean0.device
#         self.dtype  = x1_time_mean0.dtype


  
#         if noise_amp is not None:
#             print('use pre_saved noise_amp')
#             self.noise_amp  = noise_amp 
#         else:
#             print('compute noise_amp now based on current x1_time_mean0')
#             self.noise_amp = get_noise_amp(x1_time_mean0, amp_way = amp_way)
        
#         self.noise_amp = self.noise_amp.to(device = self.device, dtype = self.dtype)

#         if  white_noise: # white noise,                        with amplitude = data_mode0's amplitude
#             self.noise_amp = torch.ones_like(self.noise_amp) # * self.noise_amp[0:1, :] TODO
 

#         self.white_noise = white_noise


#         self.lam_k = self.noise_amp **2
#         self.Phi = None  

        
        

#     def sample(self, n_samples, return_hat = False, upsample=1):
#         B = n_samples
#         N = self.N  
#         C = self.C  


#         # Complex Gaussian noise with E|z|^2 = 1
#         noise = (     torch.randn(B, self.noise_amp.numel(), device=self.device, dtype=self.dtype) +
#                 1j *  torch.randn(B, self.noise_amp.numel(), device=self.device, dtype=self.dtype)) / math.sqrt(2)
#         # DC (and Nyquist if even N) must be real for irfft
#         noise[:, 0] = torch.randn(B, device=self.device, dtype=self.dtype)
#         if N % 2 == 0:
#             noise[:, -1] = torch.randn(B, device=self.device, dtype=self.dtype)
#         noise = noise.view(B, -1, C)  # (B, N//2+1, C)

#         # print(noise.shape, self.noise_amp.shape) # torch.Size([50000, 195]) torch.Size([65, 3])

#         Y = self.noise_amp * noise

#         z0 = torch.fft.irfft(Y, n=N, dim=1, norm="ortho") 


#         # print(z0.mean(), z0.std()) # tensor(1.1584e-05, device='cuda:0') tensor(0.9996, device='cuda:0')
        

#         if not return_hat:
#             return z0
#         else:
#             return Y, z0
        
    


class noise_x0_class:
    def __init__(self,
                 x1_time_mean0,
                 noise_amp=None,
                 white_noise = False,
                 amp_way  =None,
                 noise_type = 'data',   # 'data' (default), 'matern', or 'empirical_decay'
                 matern_params = None,  # dict(nu_K, ell_K, sigma2_K, M) when noise_type='matern'
                 noise_params = None,   # dict(n_empirical, eps_decay) when noise_type='empirical_decay'
                 ):

        B, N, C = x1_time_mean0.shape 
        self.N = N
        self.C = C

        self.device = "cuda:0" # x1_time_mean0.device
        self.dtype  = x1_time_mean0.dtype

        self.noise_type = noise_type

        if noise_type == 'matern':
            # ------- Matérn kernel noise -------
            assert matern_params is not None, "matern_params required when noise_type='matern'"
            nu_K    = matern_params['nu_K']
            ell_K   = matern_params['ell_K']
            sigma2_K = matern_params['sigma2_K']
            M       = matern_params['M']   # number of time points (= N above)
            assert M == N, f"matern_params['M']={M} must equal signal length N={N}"

            N_half = M // 2  # number of positive-frequency modes

            # Compute the Matérn spectral density on the FULL symmetric basis
            # ks = -N_half, ..., 0, ..., N_half  →  (2*N_half+1,) values
            # This ensures the normalization sum(lam_full) = sigma2 * M
            ks_full = torch.arange(-N_half, N_half + 1, dtype=torch.float64)
            lam_k_full = matern_lambda_periodic_from_ks(
                nu=nu_K, ell=ell_K, sigma2=sigma2_K,
                ks=ks_full, M=M,
                white=False,
                device=self.device,
                basis='fourier',
            )  # shape (2*N_half+1,)

            # Take the positive half: k = 0, 1, ..., N_half
            # Since spectrum is symmetric, lam_full[N_half + k] = lam_full[N_half - k]
            # The centre of ks_full is index N_half (corresponding to k=0)
            lam_k_half = lam_k_full[N_half:]  # shape (N_half+1,) = (M//2+1,)

            # Expand to (M//2+1, C) — same spectrum for every channel
            self.noise_amp = torch.sqrt(lam_k_half).unsqueeze(-1).expand(-1, C).clone()
            self.noise_amp = self.noise_amp.to(device=self.device, dtype=self.dtype)
            print(f'[noise_type=matern] nu_K={nu_K}, ell_K={ell_K}, sigma2_K={sigma2_K}, M={M}')
            print(f'  noise_amp shape={self.noise_amp.shape}')
            print(f'  full spectrum sum={lam_k_full.sum().item():.4f}, target={sigma2_K*M:.4f}')

        elif noise_type == 'empirical_decay':
            # ------- Empirical low modes + power-law tail -------
            if noise_params is None:
                noise_params = {}
            n_empirical = noise_params.get('n_empirical', 32)
            eps_decay   = noise_params.get('eps_decay', 0.1)

            # Compute full empirical spectrum
            empirical_amp = get_noise_amp(x1_time_mean0, amp_way=amp_way)  
            empirical_lam = empirical_amp ** 2  # variances

            # Build the hybrid spectrum
            N_freq = empirical_lam.shape[0]  # N//2+1
            lam_hybrid = empirical_lam.clone()

            # For modes j > n_empirical, scale from the last empirical mode
            # lam_k(j) = lam_k(n_empirical) * (n_empirical / j)^(1+eps)
            if n_empirical < N_freq:
                lowest_lam = empirical_lam[n_empirical]  # (C,) — value at the boundary
                js = torch.arange(n_empirical + 1, N_freq, device=empirical_lam.device).float()
                decay = (float(n_empirical) / js).unsqueeze(-1) ** (1.0 + eps_decay)  # (n_tail, 1)
                lam_hybrid[n_empirical + 1:] = lowest_lam.unsqueeze(0) * decay

            self.noise_amp = torch.sqrt(lam_hybrid)
            self.noise_amp = self.noise_amp.to(device=self.device, dtype=self.dtype)
            print(f'[noise_type=empirical_decay] n_empirical={n_empirical}, eps_decay={eps_decay}')
            print(f'  noise_amp shape={self.noise_amp.shape}')
            print(f'  trace (sum lam_k) = {lam_hybrid.sum().item():.4f}')

        else:
            # ------- Original data-driven noise -------
            if noise_amp is not None:
                print('use pre_saved noise_amp')
                self.noise_amp  = noise_amp 
            else:
                print('compute noise_amp now based on current x1_time_mean0')
                self.noise_amp = get_noise_amp(x1_time_mean0, amp_way = amp_way)
            
            self.noise_amp = self.noise_amp.to(device = self.device, dtype = self.dtype)

            if  white_noise :
                self.noise_amp = torch.ones_like(self.noise_amp) #* self.noise_amp[0:1, :]

        self.white_noise = white_noise


        self.lam_k = self.noise_amp **2
        self.Phi = None  

        
        

    def sample(self, n_samples, return_hat = False, upsample=1):
        B = n_samples
        N = self.N  
        C = self.C  


        # Complex Gaussian noise with E|z|^2 = 1
        noise = (     torch.randn(B, self.noise_amp.numel(), device=self.device, dtype=self.dtype) +
                1j *  torch.randn(B, self.noise_amp.numel(), device=self.device, dtype=self.dtype)) / math.sqrt(2)
        # DC (and Nyquist if even N) must be real for irfft
        noise[:, 0] = torch.randn(B, device=self.device, dtype=self.dtype)
        if N % 2 == 0:
            noise[:, -1] = torch.randn(B, device=self.device, dtype=self.dtype)
        noise = noise.view(B, -1, C)  # (B, N//2+1, C)

        # print(noise.shape, self.noise_amp.shape) # torch.Size([50000, 195]) torch.Size([65, 3])

        Y = self.noise_amp * noise

        z0 = torch.fft.irfft(Y, n=N, dim=1, norm="ortho") 

        if not return_hat:
            return z0
        else:
            return Y, z0
    
