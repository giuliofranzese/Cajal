import torch
import gpytorch
from typing import Optional, Tuple

def make_grid(dims, x_min=0, x_max=1):
    """ Creates a 1D or 2D grid based on the list of dimensions in dims.

    Example: dims = [64, 64] returns a grid of shape (64*64, 2)
    Example: dims = [100] returns a grid of shape (100, 1)
    """
    # if len(dims) == 1:
    assert len(dims) == 1
    grid = torch.linspace(x_min, x_max, dims[0])
    grid = grid.unsqueeze(-1)
    # elif len(dims) == 2:
    #     _, _, grid = make_2d_grid(dims)
    return grid

def _compute_and_cache_marten_cholesky(N, 
                                       bandwidth, sigma, 
                                       eps = 1e-10   #  1e-6
                                       ):
    """
    Compute and cache the Cholesky factor of the Matérn (ν=0.5) kernel covariance matrix. 
    eps = 1e-10         # Diagonal covariance jitter
    nu = 0.5            # Smoothness parameter, in [0.5, 1.5, 2.5]
    Supports both 1D (timeseries) and 2D (images) input shapes.
    Returns:
        torch.Tensor: Lower-triangular Cholesky factor of the kernel matrix.
    """
    # if self.is_unidimensional:
    # Timeseries (1D)
    
    ts = torch.linspace(0, 1, N, device='cpu')
    grid_coords = ts.unsqueeze(1)  # (N, 1)

    diff = grid_coords - grid_coords.T  # (N, N)
    dists = torch.abs(diff)  # (N, N)
    K = sigma ** 2 * torch.exp(-dists / bandwidth)
    jitter = eps * torch.eye(N, device=K.device, dtype=K.dtype)
    K = K + jitter

    L_chol = torch.linalg.cholesky(K)
    return K, L_chol


class GPPrior(gpytorch.models.ExactGP):
    """ Wrapper around some gpytorch utilities that makes prior sampling easy.
    """

    def __init__(self, kernel=None, mean=None, lengthscale=None, var=None, device='cpu'):
        """
        kernel/mean/lengthscale/var: parameters of kernel
        """
        
        # Initialize parent module; requires a likelihood so small hack
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPPrior, self).__init__(None, None, likelihood)
        
        self.device = device
        
        if mean is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = mean
        
        if kernel is None:
            eps = 1e-10         # Diagonal covariance jitter
            nu = 0.5          # Smoothness parameter, in [0.5, 1.5, 2.5]
            
            # Default settings for length/variance
            if lengthscale is None: 
                self.lengthscale = torch.tensor([0.01], device=device)
            else:
                self.lengthscale = torch.as_tensor(lengthscale, device=device)
            if var is None:
                self.outputscale = torch.tensor([0.1], device=device)   # Variance
            else:
                self.outputscale = torch.as_tensor(var, device=device)
                        
            # Create Matern kernel with appropriate lengthscale
            base_kernel = gpytorch.kernels.MaternKernel(nu,eps=eps)
            base_kernel.lengthscale = self.lengthscale
            
            # Wrap with ScaleKernel to get appropriate variance
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.covar_module.outputscale = self.outputscale
            
        else:
            self.covar_module = kernel
            
        self.eval()  # Required for sampling from prior
        if device == 'cuda':
            self.cuda()

        self.L = None  # Cache for Cholesky factor

    def check_input(self, x, dims=None):
        assert x.ndim == 2, f'Input {x.shape} should have shape (n_points, dim)'
        if dims:
            assert x.shape[1] == len(dims), f'Input {x.shape} should have shape (n_points, dim={len(dims)})'

    def forward(self, x):
        """ Creates a Normal distribution at the points in x.
        x: locations to query at, a flattened grid; tensor (n_points, dim)

        returns: a gpytorch distribution corresponding to a Gaussian at x
        """
        self.check_input(x)
        x = x.to(self.device)

        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    




    def _matern_kernel_prior(self,
        shape: Tuple[int, int, int],       # (batch, grid_points, channels)

        bandwidth, sigma, 
        eps = 1e-10   #  1e-6
    ) -> torch.Tensor:
        """
        Sample from a GP prior with Matern kernel using Cholesky decomposition.

        Args:
            shape: (batch, grid_points, channels)
            L:     Cholesky factor of kernel matrix [g, g]

        Returns:
            samples: [batch, grid_points, channels]
        """
        b, N, c = shape  # Unpack shape: b = batch size, N = number of grid points, c = channels.
        
        # Cholesky factor of the covariance matrix: [N, N]
        if (self.L is not None):
            assert (self.L.shape[0] == N)
            L = self.L
        else:
            K, L = _compute_and_cache_marten_cholesky(N, 
                                    bandwidth, sigma, 
                                    eps    #  1e-6
                                    )
            self.L = L 

        # Sample from the standard normal: shape (batch, N, channels).
        z = torch.randn(b, N, c)

        # Matrix multiply L @ z: matrix * vector
        # z: [b, N, c], L: [N, N] → output: [b, N, c]
        samples = torch.einsum('ij, bjc -> bic', L, z)

        return samples
    
    def sample(self, x, dims, n_samples=1, n_channels=1):
        """ Draws samples from the GP prior.
        x: locations to sample at, a flattened grid; tensor (n_points, n_dim)
        dims: list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
        n_samples: number of samples to draw
        n_channels: number of independent channels to draw samples for

        returns: samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        

        ####################################################################### way 1
        # x_shape = [n_samples, dims[0],  n_channels] # (b, N, c) 

        # samples = self._matern_kernel_prior(
        #     x_shape,       # (batch, grid_points, channels)
           
        #     bandwidth=0.2, sigma=0.4 , 
        #     eps = 1e-10   # default 1e-6 , fm  1e-10
        # )

        # samples = samples.permute(0, 2, 1).to(self.device)  # from (b, N, c) → (b, c, N)

        ####################################################################### way 2

        # shape = [n_samples, dims[0],  n_channels] # (b, N, c) 
        # b, N, c = shape  # Unpack shape: b = batch size, N = number of grid points, c = channels.
        
        # # Sample from the standard normal: shape (batch, N, channels).
        # z = torch.randn(b, N, c).to(self.device) 

        # # L
        # with torch.no_grad():
        #     if self.L is None:
        #         query_points = make_grid(dims).to(self.device) 
        #         K = self.covar_module(query_points, query_points).evaluate()  
        #         L = torch.linalg.cholesky(K)

        #         self.K = K
        #         self.L = L
        #     else:
        #         L = self.L

        # # Matrix multiply L @ z: matrix * vector
        # # z: [b, N, c], L: [N, N] → output: [b, N, c]
        # samples = torch.einsum('ij, bjc -> bic', L, z)

        # samples = samples.permute(0, 2, 1) # from (b, N, c) → (b, c, N)

     
        ####################################################################### way 3: functional flow matching
        self.check_input(x, dims)
        x = x.to(self.device)
        distr = self(x)
        
        samples = distr.sample(sample_shape = torch.Size([n_samples * n_channels, ]))
        samples = samples.reshape(n_samples, n_channels, *dims)

        #######################################################################
        
        return samples