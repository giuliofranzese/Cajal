import torch
import matplotlib.pyplot as plt
import sys
import copy
from neuralop.models import FNO as _FNO
import torch.nn.functional as F

import torch.nn as nn
from neuralop.layers.fourier_continuation import FCLegendre 

 
class FCWrapper(nn.Module): 
    def __init__(self, core_fno: nn.Module, n_additional_pts=32, d=4, dim=-1):
        super().__init__()
        self.core = core_fno
        self.fc = FCLegendre(d=d, n_additional_pts=n_additional_pts)
        self.dim = dim

    def forward(self, x):
        x_ext = self.fc(x, dim=self.dim)
        y_ext = self.core(x_ext)
        y = self.fc.restrict(y_ext, dim=self.dim)
        return y
    

"""
 A version of the time-conditioned FNO model.
 Uses the new neuralop package.
 Works by concatenating time as an input channel.
"""

def t_allhot(t, shape):
    batch_size = shape[0]
    n_channels = shape[1]
    dim = shape[2:]
    n_dim = len(dim)

    # t = t.view(batch_size, *[1]*(n_channels+n_dim))
    t = t.view(batch_size, *[1]*(1+n_dim)) 

    t = t * torch.ones(batch_size, 1, *dim, device=t.device)  

    return t


def make_posn_embed(batch_size, dims):
    
    if len(dims) == 1:
        # Single channel of spatial embeddings
        emb = torch.linspace(0, 1, dims[0])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(dims) == 2:
        # 2 Channels of spatial embeddings
        x1 = torch.linspace(0, 1, dims[1]).repeat(dims[0], 1).unsqueeze(0)
        x2 = torch.linspace(0, 1, dims[0]).repeat(dims[1], 1).T.unsqueeze(0)
        emb = torch.cat((x1, x2), dim=0)  # (2, dims[0], dims[1])

        # Repeat along new batch channel
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, 2, *dims)
    else:
        raise NotImplementedError
    
    return emb



class PadCropWrapper(torch.nn.Module):
    def __init__(self, core, pad_ratio=0.25, pad_mode="reflect"):
        super().__init__()
        self.core = core
        self.pad_ratio = pad_ratio
        self.pad_mode = pad_mode

    def forward(self, x):
        
        L = x.shape[-1]
        pad = int(round(self.pad_ratio * L))
        x_pad = F.pad(x, (pad, pad), mode=self.pad_mode)   
        y_pad = self.core(x_pad)
        y = y_pad[..., pad:-pad]                         
        return y


 
def even_extend(x: torch.Tensor) -> torch.Tensor:
    """
    Even extension for Neumann BC.
    x shape: (..., L)
    return shape: (..., 2*(L-1))
    """
    tail = torch.flip(x[..., 1:-1], dims=[-1])  # indices: L-2 ... 1
    return torch.cat([x, tail], dim=-1)

def crop_back(y, L):
    return y[..., :L]


class CosineFNOWrapper(torch.nn.Module):
    def __init__(self, core_fno):
        super().__init__()
        self.core = core_fno

    def forward(self, x):
      
        L = x.shape[-1]
        x_ext = even_extend(x)            # (B,C,2(L-1))
        y_ext = self.core(x_ext)          # same length
        y = crop_back(y_ext, L)      # (B,C,L)
        return y

 
 
class FNO(torch.nn.Module): 
    def __init__(self, modes, vis_channels, hidden_channels, proj_channels, x_dim=1, t_scaling=1, # 1000, 
                #  prediction='v' 
                 ): 
        super(FNO, self).__init__()

        self.model_name = 'fno'
        
        self.t_scaling = t_scaling
        
        # modes = 16
        # hidden = 32
        # proj = 64
        
        #model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        n_modes = (modes, ) * x_dim   # Same number of modes in each x dimension
        in_channels = vis_channels + x_dim + 1  # visual channels + spatial embedding + time embedding

        projection_channel_ratio =  proj_channels / hidden_channels

        # self.prediction = prediction
        
        # core 
        self.model = _FNO(n_modes=n_modes, 
                         hidden_channels=hidden_channels, 
                         projection_channel_ratio = projection_channel_ratio,
                         #  projection_channels=proj_channels,
                         in_channels=in_channels, 
                         out_channels=vis_channels,

                     
                         domain_padding=0.0, 

                         )

        
        
        
    def forward(self, t, u):
      
        
        batch_size = u.shape[0]
        dims = u.shape[2:]
        
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]
 

        # Concatenate time as a new channel
        t = t_allhot(t, u.shape) # [500, 1, 751]
        # Concatenate position as new channel(s)
        posn_emb = make_posn_embed(batch_size, dims).to(u.device) # torch.Size([500, 1, 751])

        

        u_cat = torch.cat((u, posn_emb, t), 
                      dim = 1
                      ).float() #   fix precision

        out = self.model(u_cat)

    
        return out  
    

    
 

