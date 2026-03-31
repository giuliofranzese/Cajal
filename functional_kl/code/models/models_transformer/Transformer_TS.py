import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# models.
from models.models_transformer.embedder import BottleneckPatchEmbedder, TimestepEmbedder, LabelEmbedder, TSEmbedding
from models.models_transformer.torch_models import TorchLinear, RMSNorm, SwiGLUMlp
# ts
from models.models_transformer.SelfAttention_Family import FullAttention, AttentionLayer
# img
from models.models_transformer.SelfAttention_Family import TransformerBlock

from models.models_transformer.SelfAttention_Family import fourier_features


def unsqueeze(t, dim):
    """Adds a new axis to a tensor at the given position."""
    return t.unsqueeze(dim)


#################################################################################
#                   Modern Transformer Components with Vec Gates               #
#################################################################################

class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.out_channels,
            bias=True,
            weight_init="scaled_variance", # "zeros", TODO zero then no gradient when vt from substraction
            bias_init="zeros",
        )

    def __call__(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning             #
#################################################################################


class pmfDiT(nn.Module):
    """
    MeanFlow improved Transformer (pmfDiT).
    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    def __init__(
        self,
        input_type: str = 'ts', # 'ts' for time series, 'img' for images
        input_size: int = 512,
        in_channels: int = 2,
        num_classes: int = 2,

        patch_size: int = 8,
        hidden_size: int = 768,
        depth: int = 16,

        num_heads: int = 8,
        mlp_ratio: float = 8 / 3,
        
        # aux_head_depth: int = 8,
        num_class_tokens: int = 2,
        num_time_tokens: int = 4,
        # num_cfg_tokens: int = 4,
        # num_interval_tokens: int = 2,

        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        # eval_mode: bool = False,
    ):
        """
        Set up the pmfDiT model components.
         - Patch embedder for input images.
         - Embedders for time, omega, cfg intervals, and class labels.
         - Learnable tokens for conditioning.
         - Transformer blocks with shared backbone and dual heads.
         - Final projection layers for u and v outputs.
        """
        super().__init__()
        self.model_name = 'transformer'
        self.input_type = input_type
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        # self.aux_head_depth = aux_head_depth

        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        # self.num_cfg_tokens = num_cfg_tokens
        # self.num_interval_tokens = num_interval_tokens

        self.token_init_constant = token_init_constant
        self.embedding_init_constant = embedding_init_constant
        self.weight_init_constant = weight_init_constant

        # self.eval_mode = eval_mode

        self.out_channels = self.in_channels

        self.functional = False # whether to use functional conditioning with continuous coordinate encoding. If False, use discrete positional embeddings.

        if input_type == 'img':
            self.x_embedder = BottleneckPatchEmbedder(
                self.input_size,
                self.patch_size,
                128 if self.hidden_size <= 1024 else 256, # pca channels. 256 for H/G
                self.in_channels,
                self.hidden_size,
                bias=True,
            )
        elif input_type == 'ts':
            self.x_embedder = TSEmbedding(n_vars=self.in_channels,  patch_len=self.patch_size, 
                                          input_len = self.input_size,
                                        d_model=self.hidden_size, dropout=0.0)

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
        )

        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        # self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        # self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        # self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)

        self.y_embedder = LabelEmbedder(self.num_classes, **embed_kwargs)

        token_initializer = partial(
            nn.init.normal_, std=self.token_init_constant / math.sqrt(self.hidden_size)
        )
        self.time_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_time_tokens, self.hidden_size))
        )
        self.class_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_class_tokens, self.hidden_size))
        )
        # self.omega_tokens = nn.Parameter(
        #     token_initializer(torch.empty(1, self.num_cfg_tokens, self.hidden_size))
        # )
        # self.t_min_tokens = nn.Parameter(
        #     token_initializer(torch.empty(1, self.num_interval_tokens, self.hidden_size))
        # )
        # self.t_max_tokens = nn.Parameter(
        #     token_initializer(torch.empty(1, self.num_interval_tokens, self.hidden_size))
        # )

        total_tokens = (
            self.x_embedder.num_patches
            + self.num_class_tokens
            # + self.num_cfg_tokens
            # + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.prefix_tokens = (
            self.num_class_tokens
            # + self.num_cfg_tokens
            # + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.head_dim = self.hidden_size // self.num_heads

        if input_type == 'img':
            self.register_buffer("rope_freqs", precompute_rope_freqs(   self.head_dim, self.x_embedder.num_patches))
        elif input_type == 'ts':
            self.register_buffer("rope_freqs", precompute_rope_freqs_1d(self.head_dim, self.x_embedder.num_patches))

        self.pos_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, total_tokens, self.hidden_size), std=0.02)
        )

    
        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
        )

        

        if self.functional:
            self.blocks = nn.ModuleList(
                [
                    FunctionalTransformerBlock(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        coord_dim=self.coord_dim,
                        kernel_hidden=kernel_hidden,
                    )
                    for _ in range(self.depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(
                    **block_kwargs
                 ) 
                 for _ in range(self.depth)]
            )

        self.v_final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        ) 


        
        if self.functional:
            # continuous coordinate encoding for time domain
            coord_feat_dim = (1 + 2 * self.coord_bands)
            self.coord_proj = TorchLinear(
                coord_feat_dim,
                self.hidden_size,
                bias=True,
                weight_init="scaled_variance",
                bias_init="zeros",
            )


    def _build_phys_metadata(self, batch_size: int, n_steps: int, device, dtype):
        """
        Returns:
            coords:  [batch_size, n_steps, 1] in [0, 1]
            # weights: [batch_size, n_steps] quadrature weights
        """
        if self.input_type == "ts":
            coords = torch.linspace(0.0, 1.0, steps=n_steps, device=device, dtype=dtype)
            coords = coords.view(1, n_steps, 1).expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Unknown input_type={self.input_type}")

        # # uniform quadrature weights on the source discretization
        # weights = torch.full(
        #     (batch_size, n_steps),
        #     1.0 / n_steps,
        #     device=device,
        #     dtype=dtype,
        # )

        return coords # , weights

   
    def _build_sequence(self, x, h,  y): # w, t_min, t_max,
        """
        Build the input token sequence for the transformer.
        1. Embed the input image patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input images
            h: timestep
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels

        Returns:
            seq: Token sequence for the transformer
        """

        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        # omega_embed = self.omega_embedder(1 - 1 / w)
        # t_min_embed = self.cfg_t_start_embedder(t_min)
        # t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)

        if self.functional:
            batch_size, n_steps, D = x_embed.shape
            coords = self._build_phys_metadata( 
                batch_size=batch_size,
                n_steps=n_steps,
                device=x_embed.device,
                dtype=x_embed.dtype,
            )
            # continuous coordinate encoding for time domain
            x_embed = x_embed + self.coord_proj(fourier_features(coords, self.coord_bands))


        time_tokens  = self.time_tokens  + unsqueeze(h_embed, 1) # [1, 4, 1280] + [8, 1, 1280]
        class_tokens = self.class_tokens + unsqueeze(y_embed, 1) # [1, 8, 1280]

        # print( f"time_tokens: {time_tokens.shape}, class_tokens: {class_tokens.shape}, x_embed: {x_embed.shape}" )
        # torch.Size([16, 4, 256]), class_tokens: torch.Size([16, 2, 256]), x_embed: torch.Size([16, 100, 256])
        
        # todo
        seq = torch.cat(
            [
                class_tokens,
                time_tokens,
                x_embed,
            ],
            axis=1,
        )

        if not self.functional:
            seq = seq + self.pos_embed

        return seq # [3, 70, 768]
    
    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(( x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        images = x.reshape((x.shape[0], c, h * p, w * p))
        return images


    def forward(self, x,  h,   y):
        """
        Forward pass of the pmfDiT model.
        Returns the predicted u and v components.

        Args:
            x: Input images
            t, h: time steps
            # We don't explicitly condition on time t, only on h = t - r   following https://arxiv.org/abs/2502.13129
            # w: CFG scale
            # t_min, t_max: CFG interval
            y: Class labels

        Returns:
            u: Average velocity field
            v: Instantaneous velocity field
        """

        if h.dim() == 0 or h.numel() == 1:
            h = torch.ones(x.shape[0], device=h.device) * h

        for block in self.blocks:
            if not  self.functional:
                seq = self._build_sequence(x, h, y)
            else:
                seq, coords = self._build_sequence(x, h, y)

        for block in self.blocks:
            if not  self.functional:
                seq = block(seq, self.rope_freqs)
            else:
                seq = block(seq, prefix_len=self.prefix_tokens, coords=coords)


        
 

        v_tokens = seq[:, self.prefix_tokens :]

        v = self.v_final_layer(v_tokens)


        v = v.reshape(v.shape[0], v.shape[1], self.patch_size , self.in_channels ) #  [B, seq_len] for ts
        v = v.permute(0, 3, 1, 2)
        v = v.reshape(
               (v.shape[0], # B
                v.shape[1],  # n_vars 
                v.shape[2] * v.shape[3])  # patch_num * each patch's patch_len 
               )
        
        if self.x_embedder.pad_len > 0:
            v = v[:, :, :-self.x_embedder.pad_len]

        return v


#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    dim = dim // 2 # for 2d rotary embeddings
    T = int(seq_len ** 0.5)

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(T, dtype=torch.float32)
    
    freqs_h = torch.einsum('i,j->ij', positions, freqs)
    freqs_w = torch.einsum('i,j->ij', positions, freqs)
    freqs = torch.concatenate([torch.tile(freqs_h[:, None, :], (1, T, 1)), 
                               torch.tile(freqs_w[None, :, :], (T, 1, 1))], axis=-1)  # (T, T, 2D)
    real = torch.cos(freqs).reshape(seq_len, dim)
    imag = torch.sin(freqs).reshape(seq_len, dim)
    return torch.complex(real, imag)


def precompute_rope_freqs_1d(dim: int, seq_len: int, theta: float = 10000.0):
    """
    1D RoPE for temporal sequence.
    """
     
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  

    positions = torch.arange(seq_len, dtype=torch.float32)   # [T]
    angles = torch.einsum("i,j->ij", positions, freqs)       # [T, dim/2]

    return torch.polar(torch.ones_like(angles), angles)     # complex64



#################################################################################
#                                iMF DiT Configs                                #
#################################################################################

# pmfDiT_B_16 = partial(
#     pmfDiT,  
#     input_type = 'ts',
#     input_size  = 512,
#     patch_size  = 8,
#     in_channels = 2,
#     hidden_size = 768,
#     )
# model = pmfDiT_B_16()
# x = torch.ones(3, 2, 512)
# y = torch.ones(3, ).long()
# t = torch.rand(3, )
# out = model(x, t, y)
# print(out.shape) # [48, 17, 512]


