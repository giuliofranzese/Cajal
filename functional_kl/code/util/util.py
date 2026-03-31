import torch
import numpy as np
import matplotlib.pyplot as plt
from util.util2 import MatReader
 

import yaml

import wandb
import math
import re
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
from typing import Dict, Optional

import scipy.stats as stats
import copy


import os
import random
import numpy as np


def make_grid(dims, x_min=0, x_max=1):
    """ Creates a 1D or 2D grid based on the list of dimensions in dims.

    Example: dims = [64, 64] returns a grid of shape (64*64, 2)
    Example: dims = [100] returns a grid of shape (100, 1)
    """
    if len(dims) == 1:
        grid = torch.linspace(x_min, x_max, dims[0])
        grid = grid.unsqueeze(-1)
    elif len(dims) == 2:
        _, _, grid = make_2d_grid(dims)
    return grid


def make_2d_grid(dims, x_min=0, x_max=1):
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1)),
        dim=1)
    return x1, x2, grid

def reshape_for_batchwise(x, k):
        # need to do some ugly shape-hacking here to get appropriate number of dims
        # maps tensor (n,) to (n, 1, 1, ..., 1) where there are k 1's
        return x.view(-1, *[1]*k)

def reshape_channel_last(x):
    # maps a tensor (B, C, *dims) to (B, *dims, C)
    k = x.ndim
    idx = list(range(k))
    idx.append(idx.pop(1))
    return x.permute(idx)

def reshape_channel_first(x):
    # maps a tensor (B, *dims, C) to (B, C, *dims)
    k = x.ndim
    idx = list(range(k))
    idx.insert(1, idx.pop())
    return x.permute(idx)


def load_navier_stokes(path=None, shuffle=True):
    if not path:
        path = '../data/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('u')  # (5k, 64, 64, 50)
     
    u = u.permute(0, -1, 1, 2).reshape(-1, 64, 64).unsqueeze(1)  # (25k, 1, 64, 64)
    
    if shuffle:
        idx = torch.randperm(u.shape[0])
        u = u[idx]
        
    return u


def plot_loss_curve(tr_loss, save_path, te_loss=None, te_epochs=None, logscale=True):
    fig, ax = plt.subplots()

    if logscale:
        ax.semilogy(tr_loss, label='tr')
    else:
        ax.plot(tr_loss, label='tr')
    if te_loss is not None:
        te_epochs = np.asarray(te_epochs)
        if logscale:
            ax.semilogy(te_epochs-1, te_loss, label='te')  # assume te_epochs is 1-indexed
        else:
            ax.plot(te_epochs-1, te_loss, label='te')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close(fig)

def plot_real_vs_fake(
        # x_array, 
                      y_real, y_fake, 
                      save_path=None,
                      plot_samples = 200):
    
    # x = x_array

    if torch.is_tensor(y_real):
        y_real = y_real.detach().cpu().numpy()
    if torch.is_tensor(y_fake):
        y_fake = y_fake.detach().cpu().numpy()

    M_real = y_real.shape[1]

    # rng = np.random.default_rng()  # reproducible
    # idx = rng.permutation(y_real.shape[0])  # permute row indices
    # y_real = y_real[idx]     

    x_array_real = np.arange(M_real)/M_real  # endpoint=False # [0, 1)

    M_fake = y_fake.shape[1]
    x_array_fake = np.arange(M_fake)/M_fake  # endpoint=False # [0, 1)

    plot_samples_num = min(plot_samples, min(len(y_real), len(y_fake)))  

    # # Create subplots
    # fig, axes = plt.subplots(1, 2, 
    #                         figsize=(6, 3), 
    #                         sharex=True, 
    #                         sharey=True)
    
    # # --- Left: real data
    # for i in range(plot_samples_num):
    #     axes[0].plot(y_real[i, :, 0],  y_real[i, :, 1], # x[i, :], y_real[i, :], 
    #                     color='tab:blue', alpha=0.3, linewidth=0.5)
    # axes[0].set_title(f"Real trajectories \n (model {id})")
    # axes[0].set_xlabel("x")
    # axes[0].set_ylabel("y")
    # axes[0].grid(True)

    # # --- Right: fake data
    # for i in range(plot_samples_num):
    #     axes[1].plot(y_fake[i, :, 0],  y_fake[i, :, 1], # x[i, :], y_real[i, :], 
    #                     color='tab:orange', alpha=0.3, linewidth=0.5)
    # axes[1].set_title(f"Generated trajectories \n (model {id})")
    # axes[1].set_xlabel("x")
    # axes[1].set_ylabel("y")
    # axes[1].grid(True)

    #######################################################
    D = y_real.shape[-1]

    fig, axes = plt.subplots(D, 2, figsize=(6, 3 * D),
                            sharex=True,
                            sharey=True)

    # ---- make axes always (D, 2) ----
    if D == 1:
        axes = np.array([axes])   # from (2,) -> (1, 2)

    for channel in range(D):
        # --- Left: real data
        for i in range(plot_samples_num):
            axes[channel, 0].plot(
                x_array_real,
                y_real[i, :, channel],
                color='tab:blue', alpha=0.3, linewidth=0.5
            )
        axes[channel, 0].set_title(f"Real trajectories \n (channel {channel})")
        axes[channel, 0].set_xlabel(f"x ({x_array_real.shape[0]})")
        axes[channel, 0].set_ylabel("y")
        axes[channel, 0].grid(True)

        # --- Right: fake data
        for i in range(plot_samples_num):
            axes[channel, 1].plot(
                x_array_fake,
                y_fake[i, :, channel],
                color='tab:orange', alpha=0.3, linewidth=0.5
            )
        axes[channel, 1].set_title(f"Generated trajectories \n (channel {channel})")
        axes[channel, 1].set_xlabel(f"x ({x_array_fake.shape[0]})")
        axes[channel, 1].grid(True)

    plt.suptitle(f"{D} channels")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)
 
    if save_path is not None:
        plt.savefig(save_path)
        print(f'plot saved to {save_path}')
        plt.close(fig)
    else:
        a = wandb.Image(fig)
        plt.close(fig)
        return a
                        
        # plt.show()

    #######################################################
    if D == 2:

        # Plot first N samples from each
        N = 30

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        # Real: first 30
        for i in range(min(N, y_real.shape[0])):
            x = y_real[i, :, 0]
            y = y_real[i, :, 1]
            ax[0].plot(x, y, alpha=0.5, linewidth=1)

        ax[0].set_title(f"Real (first {min(N, y_real.shape[0])})")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].axis("equal")  # optional

        # Fake: first 30
        for i in range(min(N, y_fake.shape[0])):
            x = y_fake[i, :, 0]
            y = y_fake[i, :, 1]
            ax[1].plot(x, y, alpha=0.5, linewidth=1)

        ax[1].set_title(f"Fake (first {min(N, y_fake.shape[0])})")
        ax[1].set_xlabel("x")
  

        # IMPORTANT: use adjustable='box' (works with shared axes + savefig)
        for a in ax:
            a.set_aspect('equal', adjustable='box')
        plt.tight_layout()


        
        
        if save_path is not None:

            p = save_path            # in case it's already a Path, this is fine
            out_path = p.with_name(p.stem + "_2d" + p.suffix)   # keeps same folder, adds _2d before extension

            plt.savefig(out_path, bbox_inches="tight")
 
            print(f'plot saved to {out_path}')
            plt.close(fig)
        else:
            plt.show()
 


def plot_real_vs_fake_2d(
                      y_real, y_fake, 
                      save_path=None,
                     ):
        
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    norm = plt.Normalize() # vmin=1.0, vmax=5.0
    ax[0].imshow(y_real, cmap='jet',# norm=norm
                 )
    ax[0].set_title('real')
    ax[0].axis('off')
    fig.colorbar(ax[0].images[0], ax=ax[0])

    ax[1].imshow(y_fake, cmap='jet', #norm=norm
                 )
    ax[1].set_title('fake')
    ax[1].axis('off')
    fig.colorbar(ax[1].images[0], ax=ax[1])

    # # plt.suptitle(f"{D} channels")
    # plt.tight_layout()

    # plt.subplots_adjust(wspace=0, hspace=0.1)

    if save_path is not None:
        plt.savefig(save_path)
        print(f'plot saved to {save_path}')
        plt.close(fig)
    else:
        plt.show()



def plot_samples(samples, save_path):
    n = samples.shape[0]
    sqrt_n = int(np.sqrt(n))

    fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(8,8))

    samples = samples.permute(0, 2, 3, 1)  # (b, c, h, w) --> (b, h, w, c)
    samples = samples.detach().cpu()

    for i in range(n):
        j, k = i//sqrt_n, i%sqrt_n
        
        axs[j, k].imshow(samples[i])
        
        axs[j, k].set_xticks([])
        axs[j, k].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(save_path)
    plt.close(fig)


def get_1d_plot(x:torch.Tensor,
                v:torch.Tensor,
                num_plot=1,
                xmin:float=-np.pi,
                xmax:float=np.pi,
                ymin:float=-np.pi/2.,
                ymax:float=np.pi/2.,
                sort:bool=True,
                #use_grid=False,
                origin='lower',
                figsize=(5, 5),
                color=None,
                alpha=1.0,
                plot=False,
                ):
    xsz = list(x.shape)
    batch_size, num_points, xdim = xsz[0], xsz[1], np.prod(xsz[2:])

    vsz = list(v.shape)
    b, n, vdim = vsz[0], vsz[1], np.prod(vsz[2:])
    
    assert b == batch_size
    assert (n    == num_points and vdim == 1 and xdim == 1) or \
           (vdim == xdim       and n == 1    and num_points == 1)

    x = x.view(batch_size, num_points*xdim)
    v = v.view(batch_size, num_points*vdim)

    if sort:
        v, indices = torch.sort(v, dim=1)
        x = x[torch.arange(batch_size)[:,None], indices]

    fig = plt.figure(figsize=figsize)
    for i in range(min(batch_size, num_plot)):
        if color is not None:
            plt.plot(v[i].cpu().numpy(), x[i].cpu().numpy(), alpha=alpha, color=color)
        else:
            plt.plot(v[i].cpu().numpy(), x[i].cpu().numpy(), alpha=alpha)
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    ax = plt.gca()
    ax.grid(False)

    # tight
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # close figure
        plt.close()
        return image

def sample_many(wrapper, n_samples, dims, n_channels=1, batch_size=500, save_path=None):
    n_batches = n_samples // batch_size
    n_samples = n_batches * batch_size
    print(f'Generating {n_samples} samples')


    samples = []
    generated = 0
    while generated < n_samples:
        print(f'... generated {generated}/{n_samples}')
        try:
            sample = wrapper.sample(dims, n_samples=batch_size, n_channels=n_channels)
            samples.append(sample.detach().cpu())
            del sample
            torch.cuda.empty_cache()
            generated += batch_size
        except:
            print('NaN, retry')

    samples = torch.stack(samples)

    if save_path:
        torch.save(samples, save_path)
    return samples




def save_checkpoint(epoch, model, model_optimizer, model_scheduler=None, checkpoint_file='checkpoint.pt'):
    content = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_optimizer': model_optimizer.optimizer.state_dict(),
                'model_scheduler': model_scheduler.state_dict() if model_scheduler is not None else None,
                }
    torch.save(content, checkpoint_file)





def load_checkpoint(checkpoint_file, model, model_optimizer=None, model_scheduler=None, strict=True):

    checkpoint = torch.load(checkpoint_file,  weights_only = False, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    state_dict.pop('_metadata', None)   # optional
    state_dict.pop('pos_grid', None)    # removed after resolution-invariance refactor
    model.load_state_dict(state_dict, strict=strict)

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    
    if model_optimizer is not None and 'model_optimizer' in checkpoint and checkpoint['model_optimizer'] is not None:
        model_optimizer.optimizer.load_state_dict(checkpoint['model_optimizer'])
    else:
        model_optimizer = None
    
    if model_scheduler is not None and 'model_scheduler' in checkpoint and checkpoint['model_scheduler'] is not None:
        model_scheduler.load_state_dict(checkpoint['model_scheduler'])
    else:
        model_scheduler = None 

    return model, model_optimizer, model_scheduler, epoch




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")



def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:  # noqa: E721
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser




def fmt(v):
    if isinstance(v, str):
        return v
    elif isinstance(v, float):
        return f"{v:.3f}"
    else:
        return str(v)
    




# compute spectrum
def get_Fourier_spectrum(data = None, 
                         dim=2, 
                         return_spectrum_amplitudes=False, 

                         spectrum_amplitudes = None, knrm = None
                         ):
    """
    For accuracy evaluation, we use the energy spectrum of generated samples as our criterion. 
    For a 2D sample u, the spectrum is computed as
    E(k) =  求和 k≤∥m∥2≤k+1    |uˆ(m)|2 
    where uˆ(m) denotes the Fourier coefficients of u at m ∈ Z2+*

    We compute E(k) by averaging over a sufficiently large ensemble of samples for each frequency k.

    Args:
        data (torch.Tensor): shape (num_samples, N)  (or anything where the spatial dim is last)

    Returns:
        kvals (np.ndarray): wavenumber bins (integers 0..N//2)
        Abins (np.ndarray): mean power in each |k| bin
    """
    # print( "Energy spectrum for accuracy evaluation")

    if data is not None:
    
        # FFT over the spatial dimension
        if dim == 2:
            data_hat = torch.fft.fftn(data,dim=(1,2),norm = "forward")
        elif dim == 1:
            data_hat = torch.fft.fftn(data,dim=(1,), norm = "forward")

        
        # power |x_hat|^2
        fourier_amplitudes = data_hat.abs() ** 2  
        # average over samples
        fourier_amplitudes = fourier_amplitudes.mean(dim=0).numpy() # (N,)
        if return_spectrum_amplitudes:
            spectrum_amplitudes = copy.deepcopy(fourier_amplitudes)
        fourier_amplitudes = fourier_amplitudes.flatten()


        # integer frequency index in "cycles per box" units (same as your * npix)
        npix = data_hat.shape[-1]
        kfreq = np.fft.fftfreq(npix) * npix
        if dim == 2:
            kfreq = np.meshgrid(kfreq, kfreq)
        if dim == 2:
            knrm = np.sqrt(kfreq[0]**2 + kfreq[1]**2)
        elif dim == 1:
            knrm = np.abs(kfreq)      
        if return_spectrum_amplitudes:
            knrm_return = copy.deepcopy(knrm)
        knrm = knrm.flatten()

    elif spectrum_amplitudes is not None and knrm is not None:
        fourier_amplitudes = spectrum_amplitudes.flatten()
        knrm = knrm.flatten()
        npix = int( np.max(knrm)*2 )  # assuming knrm goes up to Nyquist

    else:
        raise ValueError("Either data or (spectrum_amplitudes, knrm) must be provided.")
        
    
   
    kbins = np.arange(0.5, npix//2+1, 1.)  # [0.5, 1.5, 2.5, ...,    time_grid_size//2+0.5]
 
    kvals = 0.5 * (kbins[1:] + kbins[:-1]) # [   1,   2,   3,..., time_grid_size//2=N] for Nyquist condition

    
    Abins_w, _, _ = stats.binned_statistic(x=knrm, # frequencies 
                                     values=fourier_amplitudes, # energy spectrum
                                     statistic = "mean", # compute the mean of values for points within each bin. 
                                     bins = kbins) 
    

    
    if dim == 2:
        counts = np.pi * (kbins[1:]**2 - kbins[:-1]**2) 
    elif dim == 1:
        '''
        for most k>0, you have two modes: +k and -k → should be 2
        for k=0, only one mode → 1 (often excluded)
        for Nyquist (when N even), only one mode → 1
        '''
       

        
        counts = 2.0 * np.ones_like(kvals, dtype=np.float64)
     
        if npix % 2 == 0:
            counts[-1] = 1.0

   

    Abins_w     *=  counts

    if return_spectrum_amplitudes:
        return kvals, Abins_w,   spectrum_amplitudes, knrm_return
    else:
        return kvals, Abins_w
    



def plot_spectrum_comparison(x1_time, x1_time_gen, x0_time, 
                             x1_time_A = None, 
                             save_path=None, wandb_img=True):

    try:
        x1_time = x1_time.detach().cpu() 
        x1_time_gen = x1_time_gen.detach().cpu() 
        x0_time = x0_time.detach().cpu() 

        if x1_time_A is not None:
            x1_time_A = x1_time_A.detach().cpu() 
        
    except:
        pass

    B, N, C = x1_time.shape 

    fig, axes = plt.subplots(
        nrows=C, ncols=1,
        figsize=(5, 4 * C),
        sharex=True, sharey=True
    )

    # If C == 1, axes is not an array
    if C == 1:
        axes = [axes]

    from kl_functions_torch import get_noise_amp
    Abins_w1_gen_allc = get_noise_amp(x1_time_gen, amp_way=None )
    Abins_w1_allc = get_noise_amp(x1_time, amp_way=None )
    Abins_w0_allc = get_noise_amp(x0_time, amp_way=None )

 
    if x1_time_A is not None:
        Abins_w1_A_allc = get_noise_amp(x1_time_A, amp_way=None )

    nnn = Abins_w1_gen_allc.shape[0]
 
    for c, ax in enumerate(axes):
  

        ax.plot(np.arange(1, nnn+1),  Abins_w1_gen_allc[:, c], linewidth=2, marker='o', markersize=6, markevery=2, label=f'gen x1 ({x1_time_gen.shape[0]} samples)')
        ax.plot(np.arange(1, nnn+1),  Abins_w1_allc[:, c],     linewidth=2, marker='o', markersize=6, markevery=2, label=f'truth x1 ({x1_time.shape[0]} samples)')
        ax.plot(np.arange(1, nnn+1),  Abins_w0_allc[:, c],     linewidth=2, marker='o', markersize=6, markevery=2, label=f'noise x0 ({x0_time.shape[0]} samples)')


        if x1_time_A is not None:
            Abins_w1_diff = np.abs( Abins_w1_allc[:, c] - Abins_w1_A_allc[:, c] ).sum().item()
            ax.plot(np.arange(1, nnn+1),   Abins_w1_A_allc[:, c],     linewidth=2, marker='o', markersize=6, markevery=2, 
                    label=f'truth x1_A ({x1_time.shape[0]} samples, diff2A={Abins_w1_diff:.2f})')
            

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f"Energy Spectrum — channel {c}")
     

        ax.grid(True, which="both", alpha=0.25)

    # Common labels (since sharex/sharey=True)
    axes[-1].set_xlabel(f'Wavenumber k (N={nnn}), idx1=DC)')
     

    # One legend for the whole figure (cleaner)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels,
           loc="upper center",
           ncol=1,                 
           frameon=False)
    
    

    fig.tight_layout()
 
    # print(f"{C} channels")
    if save_path is not None:
        out_path = save_path
        plt.savefig(out_path)
        print(f"plot saved to {out_path}")
        plt.close(fig)
    else:
        if wandb_img:
            a = wandb.Image(fig)
            plt.close(fig)
            return a
        else:
            plt.show()

 



def seed_everything(seed: int) -> None:
    # Python built-in RNG
    random.seed(seed)

    # Hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    try:
        import torch

        # PyTorch RNGs
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For newer PyTorch versions (1.8+)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass





def parse_key_token(token, current_dict=None):
    """Convert path token to int when appropriate."""
    if current_dict and isinstance(current_dict, dict):
        # If current dict already has an int key matching this token, use int
        if token.isdigit():
            int_token = int(token)
            if int_token in current_dict:
                return int_token

    # Fallback: numeric path parts become ints
    if token.isdigit():
        return int(token)

    return token

def set_nested_value(d, key_path, value):
    """
    Set nested dict value using dot notation.
    Example:
      data_params.X1.A_path.401=../data/new.npy
    """
    tokens = key_path.split(".")
    cur = d

    for raw_token in tokens[:-1]:
        token = parse_key_token(raw_token, cur)

        if token not in cur or not isinstance(cur[token], dict):
            cur[token] = {}
        cur = cur[token]

    last_token = parse_key_token(tokens[-1], cur)
    cur[last_token] = value

def apply_overrides(config, overrides):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use KEY=VALUE format.")

        key, value = item.split("=", 1)
        parsed_value = yaml.safe_load(value)
        set_nested_value(config, key, parsed_value)

    return config
