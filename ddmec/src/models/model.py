import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, t):
        # t: [batch, 1]
        return self.net(t)

class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # for layer in self.net:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.zeros_(layer.weight)
        #         nn.init.zeros_(layer.bias)
    
    def forward(self, c):
        # c: [batch, cond_dim]
        return self.net(c)

class FiLM(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)
    
    def forward(self, x, cond):
        # x: [batch, feature_dim]
        # cond: [batch, cond_dim]
        scale = self.scale(cond)  # [batch, feature_dim]
        shift = self.shift(cond)
        return x * (1 + scale) + shift

class ResidualBlock(nn.Module):
    def __init__(self, feature_dim, cond_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        # FiLM conditioning expects a conditioning vector of dimension cond_dim.
        self.film = FiLM(feature_dim, cond_dim)
    
    def forward(self, x, cond):
        # x: [batch, feature_dim]
        # cond: [batch, cond_dim]
        residual = x
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        # Apply FiLM modulation using the combined condition embedding
        h = self.film(h, cond)
        h = self.dropout(h)
        return residual + h


def sinusoidal_positional_encoding(t, embedding_dim):
    """
    Sinusoidal positional encoding for time step `t`.
    Args:
        t (int or tensor): The timestep, which could be a scalar or a tensor.
        embedding_dim (int): The dimension of the embedding.
    Returns:
        torch.Tensor: The sinusoidal position encoding for timestep `t`.
    """
    # Create the time step tensor
    t = t.float().view(t.shape[0],)
    
    # Define the position indices for sinusoidal functions
    half_dim = embedding_dim // 2
    freqs = 10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(t.device)
    
    # Sinusoidal functions for position encoding
    angle = t.unsqueeze(-1) / freqs
    sin_encoding = torch.sin(angle)
    cos_encoding = torch.cos(angle)
    
    # Concatenate sine and cosine components
    positional_encoding = torch.cat([sin_encoding, cos_encoding], dim=-1)
    
    return positional_encoding

class Denoiser_oa(nn.Module):
    def __init__(self, input_dim=50, cond_dim=50, feature_dim=128, embed_dim=128, num_blocks=4, dropout_rate=0.0):
        """
        input_dim: dimension of RNA representation (50)
        cond_dim: dimension of ATAC vector (50)
        feature_dim: hidden dimension used in the MLP blocks
        embed_dim: embedding dimension for time and condition embeddings
        num_blocks: number of residual blocks
        dropout_rate: dropout rate for regularization in the residual blocks
        """
        super().__init__()
        self.input_fc = nn.Linear(input_dim, feature_dim)
        #self.time_embed = TimeEmbedding(embed_dim)
        self.time_embed_sinus = lambda t: sinusoidal_positional_encoding(t, embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)
        self.cond_embed = ConditionEmbedding(cond_dim, embed_dim)
        # Each block expects a conditioning vector that is the concatenation of time and condition embeddings.
        self.blocks = nn.ModuleList([
            ResidualBlock(feature_dim, embed_dim * 2, dropout_rate=dropout_rate) 
            for _ in range(num_blocks)
        ])
        self.output_fc = nn.Linear(feature_dim, input_dim)
    
    def forward(self, x, condition=None, timestep=None, drop=0.0, cfg=False):
        """
        x: noisy RNA vector [batch, 50]
        condition: ATAC vector [batch, 50] (or None for unconditional generation)
        timestep: time step tensor [batch] or [batch, 1]
        drop: probability of dropping the condition (simulating classifier-free guidance)
        cfg: flag for classifier-free guidance (not used explicitly here)
        """
        # Debug prints (can be removed)
     #   print("x", x.shape)
    #     if condition is not None:
  #      print("condition", condition.shape)
    #     else:
   #      print("condition is None")
        
        # Project input RNA vector to feature space
        x = self.input_fc(x.float())
        
        # Embed the timestep
        t_emb = self.time_embed_sinus(timestep)
        t_emb = self.time_embed(t_emb)
        
        # Process the conditional ATAC vector
        if condition is None:
            # Use a zero tensor as the null embedding for unconditional generation
            c_emb = torch.zeros_like(t_emb)
        else:
            c_emb = self.cond_embed(condition.float())
            # Apply dropout on the condition if drop > 0
            if drop > 0:
                mask = torch.tensor(np.random.choice([0, 1], size=(c_emb.shape[0],), p=[drop, 1-drop]),
                                    device=c_emb.device)
                mask = mask.view(mask.shape[0], *([1] * (c_emb.ndim - 1)))
                c_emb = c_emb * mask
        
        # Combine time and condition embeddings via concatenation
        cond_embed = torch.cat([t_emb, c_emb], dim=-1)
    #    print("cond", cond_embed.shape)
        
        # Pass through the residual blocks
        for block in self.blocks:
            x = block(x, cond_embed)
        
        # Project back to input dimension
        out = self.output_fc(x)
     #   print("out", out.shape)
        return out
