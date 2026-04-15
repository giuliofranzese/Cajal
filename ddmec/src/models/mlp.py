import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingModule(nn.Module):
    def __init__(self, num_categories, embedding_dim, normalize=False):
       
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        self.num_categories =num_categories
        self.normalize= normalize
        
    def forward(self, x):
        if self.normalize:
            return x/self.num_categories
        else:
            return self.embedding(x)


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
    


class Denoiser(nn.Module):
    def __init__(self, input_dim, cond_dim, embedding_dim, hidden_dim, drop_prob=0.0,size="m"):
        super(Denoiser, self).__init__()
        self.time_encoding = lambda t: sinusoidal_positional_encoding(t, embedding_dim)
        #self.time_encoding =nn.Linear(1, embedding_dim)
        self.embed = nn.Linear(input_dim, embedding_dim)
        self.embed_c = nn.Linear(cond_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.size = size
        if self.size =="m":
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)


    def forward(self, x, condition=None, timestep=None, drop=0.0, cfg=False):
        
        if cfg:
            x = torch.cat([x, x], dim=0)
            condition = torch.cat([condition, torch.zeros_like(condition)], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            out = self.forward_(x, condition, timestep, drop)
            cond, uncond = torch.split(out, x.shape[0] // 2, dim=0)
            return cond, uncond
        else:
            return self.forward_(x, condition, timestep, drop)

    def forward_(self, x,condition=None, timestep=None,drop=0.0 ):
    
        x_emb = self.embed(x.float())  # (B, input_dim, embedding_dim)
        timestep =timestep.view(-1,1).float()
  
        t_emb = self.time_encoding(timestep)  # (B, embedding_dim)
        
        if condition is not None:
            c_embed =self.embed_c(condition.float())
            if drop>0:
                mask = torch.tensor( np.random.choice([0, 1], size=(c_embed.shape[0],), p=[drop,1-drop]) ,device = c_embed.device)
                mask = mask.view(mask.shape[0], *([1] * (c_embed.ndim - 1)))
                c_embed = c_embed * mask
            h = torch.cat([x_emb, t_emb +c_embed],  dim=-1)  # (B, input_dim * embedding_dim + embedding_dim + cond_emb_dim)
            
            #h = torch.cat([x_emb, t_emb ,c_embed],  dim=-1)
        else:
            h = torch.cat([x_emb,t_emb + torch.zeros_like(t_emb) ], dim=-1)  # (B, input_dim * embedding_dim + embedding_dim)

    
        h = F.relu(self.fc1(h))  # (B, hidden_dim)
        h = F.relu(self.fc2(h))  # (B, hidden_dim)
        if self.size =="m":
            h = F.relu(self.fc3(h))  # (B, hidden_dim)
            h = F.relu(self.fc4(h))  # (B, hidden_dim)

        embeddings = self.output_layer(h)  # (B, embedding_dim)
        return embeddings




class Denoiser2(nn.Module):
    def __init__(self, input_dim, cond_dim, embedding_dim, hidden_dim, drop_prob=0.0):
        super(Denoiser, self).__init__()
        self.time_encoding = lambda t: sinusoidal_positional_encoding(t, embedding_dim)
        self.time_mlp= nn.Sequential(
            self.time_encoding,
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim))

        self.embed = nn.Sequential(
             nn.Linear(input_dim, embedding_dim),
             nn.ReLU(),
             nn.Linear(embedding_dim, embedding_dim))


        self.embed_c =nn.Sequential( nn.Linear(cond_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding))


        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)


    def forward(self, x, condition=None, timestep=None, drop=0.0, cfg=False):
        
        if cfg:
            x = torch.cat([x, x], dim=0)
            condition = torch.cat([condition, torch.zeros_like(condition)], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            out = self.forward_(x, condition, timestep, drop)
            cond, uncond = torch.split(out, x.shape[0] // 2, dim=0)
            return cond, uncond
        else:
            return self.forward_(x, condition, timestep, drop)

    def forward_(self, x,condition=None, timestep=None,drop=0.0 ):
        
        x_emb = self.embed(x.float())  # (B, input_dim, embedding_dim)
        timestep =timestep.view(-1,1).float()
  
        t_emb = self.time_encoding(timestep)  # (B, embedding_dim)
        
        if condition is not None:
            c_embed =self.embed_c(condition.float())
            if drop>0:
                mask = torch.tensor( np.random.choice([0, 1], size=(c_embed.shape[0],), p=[drop,1-drop ]) ,device = c_embed.device)
                mask = mask.view(mask.shape[0], *([1] * (c_embed.ndim - 1)))
                c_embed = c_embed * mask
            h = torch.cat([x_emb, t_emb +c_embed],  dim=-1)  # (B, input_dim * embedding_dim + embedding_dim + cond_emb_dim)
        else:
            h = torch.cat([x_emb,t_emb ], dim=-1)  # (B, input_dim * embedding_dim + embedding_dim)

    
        h = F.relu(self.fc1(h))  # (B, hidden_dim)
        h = F.relu(self.fc2(h))  # (B, hidden_dim)
        h = F.relu(self.fc3(h))  # (B, hidden_dim)
        h = F.relu(self.fc4(h))  # (B, hidden_dim)

        out = self.output_layer(h)  # (B, embedding_dim)
        return out



    
