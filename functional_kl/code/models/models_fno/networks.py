 
import numpy as np
import torch
from torch.nn.functional import silu

from neuralop.layers.spectral_convolution import SpectralConv

from torch.profiler import profile, ProfilerActivity

import sys
sys.path.append('../')
from torch_utils import persistence

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.t()
        if self.bias is not None:
            x = x.add_(self.bias)
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


@persistence.persistent_class
class Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, ],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):  
        assert not (up and down)
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel , fan_out=out_channels * kernel )
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel,], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0) / f.sum().square()

        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight if self.weight is not None else None
        b = self.bias if self.bias is not None else None

        f = self.resample_filter if self.resample_filter is not None else None
        w_pad =  w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose1d(x, f.mul(4).tile([self.in_channels, 1,  1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv1d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv1d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv1d(x, f.tile([self.out_channels, 1,  1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose1d(x, f.mul(4).tile([self.in_channels, 1,  1]), groups=self.in_channels, stride=2, padding=f_pad,
                                                         output_padding=1,  ) # 我加的
            if self.down:
                x = torch.nn.functional.conv1d(          x, f.tile(       [self.in_channels, 1,  1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv1d(x, w, padding=w_pad)
        if b is not None:
            x = x + b.reshape(1, -1,  1)
        return x


# ----------------------------------------------------------------------------
# Group normalization.


@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=8, #32, 
               min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps)
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

 
# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

 
# ----------------------------------------------------------------------------


@persistence.persistent_class
class UNOBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        n_modes,
        rank=1.0,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0.0,
        skip_scale=1,
        eps=1e-5,
        group_norm=True,
        resample_filter=[1,],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else (num_heads if num_heads is not None else out_channels // channels_per_head)
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        if up and down:
            raise ValueError("Only one of 'up' and 'down' can be True.")
        resolution_scaling_factor = 1
        if up:
            resolution_scaling_factor = 2
        if down:
            resolution_scaling_factor = 0.5

        if group_norm:
            self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        else:
            self.norm0 = torch.nn.Identity()

        self.conv0 = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_scaling_factor=resolution_scaling_factor,
            n_modes=n_modes,
            rank=rank,
            factorization="ComplexTucker" if rank < 1.0 else None,
            implementation= "reconstructed", # "factorized",
        )

        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)

        if group_norm:
            self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        else:
            self.norm1 = torch.nn.Identity()

        self.conv1 = SpectralConv(
            in_channels=out_channels,
            out_channels=out_channels,
            resolution_scaling_factor=1,
            n_modes=n_modes,
            rank=rank,
            factorization="ComplexTucker" if rank < 1.0 else None,
            implementation="factorized",
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, resample_filter=resample_filter, up=up, down=down, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv1d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x + params))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] *  self.num_heads, 
                                                      x.shape[1] // self.num_heads, 
                                                      3, 
                                                      -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = x + self.proj(a.reshape(*x.shape))
            x = x * self.skip_scale
        return x



 
 
@persistence.persistent_class
class SongUNO(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of color channels at y.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        fmult= 0.25, # 0.5 ,
        rank= 1.0, # 0.1 ,
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 2],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.x
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # [8] List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[1, ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        disable_skip=False,  # Disable skip connections?
        cond=False,  # Conditional or unconditional model?
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        assert augment_dim == 0, "Augmentation regularisation is not implemented"
        assert cond is False, "Conditional model is not implemented"

        super().__init__()

        self.label_dropout = label_dropout
        self.disable_skip = disable_skip
        self.cond = cond

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == "positional" else FourierEmbedding(num_channels=noise_channels)
        # self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            n_modes = int(res * fmult)
            print('res:', res, 'n_modes:', n_modes)

            if level == 0:
                cin = cout
                cout = model_channels
                cy = label_dim
                # lifting channel by 1x1 kernel
                if cond:
                    self.enc[f"{res}x{res}_conv"] = Conv1d(in_channels=cin + cy + 1, out_channels=cout, kernel=1, **init)  # kernel=3
                else:
                    self.enc[f"{res}x{res}_conv"] = Conv1d(in_channels=cin + 1, out_channels=cout, kernel=1, **init)
                # self.enc[f'{res}x{res}_conv'] = UNOBlock(in_channels=cin+cy, out_channels=cout, n_modes=(n_modes_max, n_modes_max), rank=rank, group_norm=False, **block_kwargs)
            else:
                self.enc[f"{res}x{res}_down"] = UNOBlock(in_channels=cout, out_channels=cout, n_modes=(n_modes,), rank=rank, down=True, **block_kwargs)
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv1d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f"{res}x{res}_aux_skip"] = Conv1d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv1d(in_channels=caux, out_channels=cout, kernel=1, down=True, resample_filter=resample_filter, fused_resample=True, **init)  # kernel=3
                    # self.enc[f'{res}x{res}_aux_residual'] = UNOBlock(in_channels=caux, out_channels=cout, n_modes=(n_modes,n_modes), down=True, **block_kwargs)
                    caux = cout

            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNOBlock(in_channels=cin, out_channels=cout, n_modes=(n_modes,), rank=rank, attention=attn, **block_kwargs)

        if not self.disable_skip:
            skips = [block.out_channels for name, block in self.enc.items() if "aux" not in name]
        else:
            skips = [0 for name, block in self.enc.items() if "aux" not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            n_modes = int(res * fmult)
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNOBlock(in_channels=cout, out_channels=cout, n_modes=(n_modes,), rank=rank, attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = UNOBlock(in_channels=cout, out_channels=cout, n_modes=(n_modes,), rank=rank, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNOBlock(in_channels=cout, out_channels=cout, n_modes=(n_modes,), rank=rank, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNOBlock(in_channels=cin, out_channels=cout, n_modes=(n_modes,), rank=rank, attention=attn, **block_kwargs)
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                # Had issus when making this a spectralconv, just make it a 1x1 kernel.
                self.dec[f"{res}x{res}_aux_conv"] = Conv1d(in_channels=cout, out_channels=out_channels, kernel=1, **init_zero)  # kernel=3

    def forward(self,  noise_labels,  x, 
                class_labels=None, augment_labels=None, grid=None):
        # Mapping.
        x = x.to(noise_labels.dtype)
        
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if augment_labels is not None and self.map_augment is not None:
            emb = emb + self.map_augment(augment_labels)

        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x

        if grid is None:
            grid = self.get_grid(x.shape, x.device)

        # class_labels = self.map_label(class_labels)
        if self.cond:
            x = torch.cat((x, class_labels, grid), dim=1)
        else:
            x = torch.cat((x, grid), dim=1)

        for b, (name, block) in enumerate(self.enc.items()):
            
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
                if self.disable_skip:
                    skips[-1] = torch.zeros_like(skips[-1])
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
                if self.disable_skip:
                    skips[-1] = torch.zeros_like(skips[-1])
            else:
                
                x = block(x, emb) if isinstance(block, UNOBlock) else block(x)
                if not self.disable_skip:
                    skips.append(x)
                else:
                    skips.append(torch.zeros_like(x))

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

    # def get_grid(self, shape, device):
    #     batchsize, size_x,   = shape[0], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
    #     return gridx.to(device)

    def get_grid(self, shape, device):
        B, L = shape[0], shape[2]
        if (not hasattr(self, "_grid_cache")
            or self._grid_cache.device != device
            or self._grid_cache.shape[-1] != L):
            self._grid_cache = torch.linspace(0, 1, L, device=device).view(1, 1, L)
        return self._grid_cache.expand(B, -1, -1)

 
 