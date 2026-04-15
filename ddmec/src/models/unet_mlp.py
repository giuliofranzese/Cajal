import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TimeEmbedding, ConditionEmbedding, FiLM


class UnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        if cond_dim is not None:
            self.film = FiLM(out_dim, cond_dim)
        else:
            self.film = None

    def forward(self, x, cond=None):
        h = self.fc(x)
        h = self.norm(h)
        h = self.act(h)
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        return h


class UnetMLP_simple(nn.Module):
    """A compact U-Net-like MLP with skip connections.

    Designed to operate on flat feature vectors (e.g., gene expression).
    """
    def __init__(self, input_dim=50, cond_dim=50, base_dim=64, depth=3, embed_dim=128):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, base_dim)
        self.time_embed = TimeEmbedding(embed_dim)
        self.cond_embed = ConditionEmbedding(cond_dim, embed_dim)

        # encoder
        dims = [base_dim * (2 ** i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        in_d = base_dim
        for d in dims:
            self.enc_blocks.append(UnetBlock(in_d, d, cond_dim=embed_dim * 2))
            in_d = d

        # bottleneck
        self.bottleneck = UnetBlock(in_d, in_d * 2, cond_dim=embed_dim * 2)

        # decoder
        self.dec_blocks = nn.ModuleList()
        rev_dims = list(reversed(dims))
        in_d = in_d * 2
        for d in rev_dims:
            self.dec_blocks.append(UnetBlock(in_d + d, d, cond_dim=embed_dim * 2))
            in_d = d

        self.output_fc = nn.Linear(base_dim, input_dim)

    def forward(self, x, cond=None, t=None):
        # initial projection
        h = self.input_fc(x.float())

        if t is None:
            t = torch.zeros(h.shape[0], 1, device=h.device)
        t_emb = self.time_embed(t.float())
        if cond is None:
            c_emb = torch.zeros_like(t_emb)
        else:
            c_emb = self.cond_embed(cond.float())

        cond_comb = torch.cat([t_emb, c_emb], dim=-1)

        # encoder pass, store skips
        skips = []
        for enc in self.enc_blocks:
            h = enc(h, cond_comb)
            skips.append(h)

        # bottleneck
        h = self.bottleneck(h, cond_comb)

        # decoder pass, concatenating skips
        for dec in self.dec_blocks:
            skip = skips.pop()
            # concat along feature dim
            h = torch.cat([h, skip], dim=-1)
            h = dec(h, cond_comb)

        # project back
        # ensure shape matches base_dim
        if h.shape[-1] != self.output_fc.in_features:
            # reduce via linear
            h = nn.Linear(h.shape[-1], self.output_fc.in_features).to(h.device)(h)

        out = self.output_fc(h)
        return out
