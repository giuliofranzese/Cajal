import torch
from torch import nn
from .conditioner_timestep import ConditionerTimestep


class ConditionerTimestepAndClass(nn.Module):
    """Extends ConditionerTimestep with class-label conditioning.

    Output = timestep_embedding + class_embedding, both mapped to cond_dim = dim * 4.
    This plugs into the DitPerceiverBlock's `cond` kwarg in encoder/decoder.
    """

    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.timestep_conditioner = ConditionerTimestep(dim)
        cond_dim = dim * 4
        self.class_embed = nn.Embedding(num_classes, cond_dim)

    def forward(self, timestep, class_label):
        """
        Args:
            timestep: [B] float tensor of diffusion timesteps
            class_label: [B] long tensor of class labels (0 or 1)
        Returns:
            [B, cond_dim] conditioning vector
        """
        t_emb = self.timestep_conditioner(timestep)  # [B, cond_dim]
        c_emb = self.class_embed(class_label)          # [B, cond_dim]
        return t_emb + c_emb
