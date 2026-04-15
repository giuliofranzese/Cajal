import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings as in Vaswani et al. "Attention is All You Need".
    timesteps: tensor of shape (batch,) floats or ints
    returns: (batch, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.unsqueeze(1).float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalDenoiserMLP(nn.Module):
    def __init__(self, input_dim, cond_dim, embedding_dim, hidden_dim, cond_method='concat'):
        super().__init__()
        assert cond_method in ['concat', 'film', 'adanorm'], \
            "cond_method must be one of ['concat','film','adanorm']"
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cond_method = cond_method

        # time embedding: sinusoidal -> linear
        self.time_proj = nn.Linear(embedding_dim, hidden_dim)

        # condition embedding (for film or adanorm)
       # if cond_method in ['film', 'adanorm']:
        layers = [
                nn.Linear(cond_dim, hidden_dim ),
                nn.SiLU(),
                nn.Linear(hidden_dim , hidden_dim if cond_method == 'concat' else hidden_dim*2)
        ]
        self.cond_mlp = nn.Sequential(*layers)
      
        # LayerNorm for hidden layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        if cond_method == 'adanorm':
            # separate norm before FiLM
            self.cond_norm = nn.LayerNorm(hidden_dim)

        # MLP layers
        # in_dim = input_dim + hidden_dim if cond_method != 'concat' else input_dim + cond_dim + hidden_dim

        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim *3 if cond_method == 'concat' else hidden_dim *2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # project back to input_dim for skip connection
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, condition=None, timestep=None, drop=0.0,cfg=None):
        """
        x: tensor of shape (batch, input_dim)
        condition: tensor of shape (batch, cond_dim) or None
        timestep: tensor of shape (batch,) or None
        drop: fraction of samples in batch to drop condition
        """
        x = x.float()
        batch_size = x.size(0)
        # prepare condition
        if condition is None:
            cond = torch.zeros(batch_size, self.cond_dim, device=x.device)
        else:
            cond = condition.float()
        if drop > 0:
            mask = (torch.rand(batch_size, device=x.device) > drop).float().unsqueeze(-1)
            cond = cond * mask
        

        # sinusoidal time embedding
        if timestep is None:
            t_emb = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            sin_emb = sinusoidal_time_embedding(timestep, self.embedding_dim)
            t_emb = self.time_proj(sin_emb)

        x = self.in_layer(x)
        cond = self.cond_mlp(cond)
        # initial feature concat
        if self.cond_method == 'concat':
            h = torch.cat([x, cond, t_emb], dim=-1)
        else:
            h = torch.cat([x, t_emb], dim=-1)
 
        h1 = self.fc1(h)
        h1 = self.norm1(h1)
        h1 = F.silu(h1)

        # apply conditioning
        if self.cond_method == 'film':
            gamma, beta = cond.chunk(2, dim=-1)
            h1 = gamma * h1 + beta
        elif self.cond_method == 'adanorm':

            gamma, beta = cond.chunk(2, dim=-1)
            h1 = gamma * h1 + beta
       
    
        h2 = self.fc2(h1)
        h2 = self.norm2(h2)
        h2 = F.silu(h2 + h1)

        h3 = self.fc3(h2)
        h3 = self.norm3(h3)
        h4 = F.silu(h3 + h2)
        # output projection + skip connection
        return self.out(h4) 


if __name__ == '__main__':
    # Basic conditioning check
    torch.manual_seed(0)
    batch, D = 4, 10
    x = torch.randn(batch, D)
    y_zero = torch.zeros(batch, D)
    y_rand = torch.randn(batch, D)
    t = torch.randint(0, 1000, (batch,))

    for method in ['concat', 'film', 'adanorm']:
        model = ConditionalDenoiserMLP(D, D, embedding_dim=16, hidden_dim=64, cond_method=method)
        model.eval()
        out_zero = model.forward(x, condition=y_zero, timestep=t,drop=0.5)
        out_rand = model.forward(x, condition=y_rand, timestep=t,drop=0.5)
        diff = (out_rand - out_zero).abs().mean().item()
        print(f"Method={method}: avg |f(x,y_rand)-f(x,0)| = {diff:.6f}")
