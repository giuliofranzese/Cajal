import einops
import torch
from torch import nn

from .conditioner_timestep_class import ConditionerTimestepAndClass
from .encoder_supernodes import EncoderSupernodes
from .decoder_perceiver import DecoderPerceiver


class MINOT_TS(nn.Module):
    """MINO-T adapted for 1D time-series flow matching in FFM_KL.

    Interface:
        forward(x, h, y) -> [B, D, T]
        where x: [B, D, T] noisy input, h: [B] timestep, y: [B] class label
    """

    model_name = 'mino_t'

    def __init__(
        self,
        input_size,      # T (sequence length, e.g. 256)
        in_channels,     # D (number of feature channels)
        num_classes=2,
        radius=0.05,
        enc_dim=128,
        enc_depth=3,
        enc_num_heads=8,
        dec_dim=128,
        dec_depth=2,
        dec_num_heads=8,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels

        # Conditioner: timestep + class label -> cond_dim
        cond_dim = enc_dim * 4
        self.conditioner = ConditionerTimestepAndClass(dim=enc_dim, num_classes=num_classes)

        # Encoder: GNO supernode pooling + perceiver blocks
        self.encoder = EncoderSupernodes(
            input_dim=in_channels,
            ndim=1,              # 1D spatial dimension
            radius=radius,
            enc_dim=enc_dim,
            enc_depth=enc_depth,
            enc_num_heads=enc_num_heads,
            cond_dim=cond_dim,
        )

        # Decoder: perceiver cross-attention blocks
        self.decoder = DecoderPerceiver(
            input_dim=enc_dim,
            output_dim=in_channels,
            ndim=1,
            dim=dec_dim,
            depth=dec_depth,
            num_heads=dec_num_heads,
            cond_dim=cond_dim,
        )

        # No fixed position grid — computed on-the-fly for resolution invariance

    def forward(self, x, h, y):
        """
        Args:
            x: [B, D, T] noisy input features
            h: [B] diffusion timestep (float)
            y: [B] class label (long)
        Returns:
            pred: [B, D, T] predicted velocity / output
        """
        batch_size = x.shape[0]

        # Handle scalar timestep from ODE integrator
        if h.dim() == 0:
            h = h.unsqueeze(0).expand(batch_size)

        # Conditioning
        condition = self.conditioner(h, y)  # [B, cond_dim]

        # Build position grid on-the-fly from actual input length
        T = x.shape[-1]
        input_pos = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).expand(batch_size, -1, -1)

        # Rearrange to MINO convention: [B, T, dim]
        input_pos_bt1 = einops.rearrange(input_pos, 'b d t -> b t d')   # [B, T, 1]
        input_feat_btd = einops.rearrange(x, 'b d t -> b t d')          # [B, T, D]
        query_pos_bt1 = input_pos_bt1  # same resolution in/out

        # Encoder
        latent = self.encoder(
            input_feat=input_feat_btd,
            input_pos=input_pos_bt1,
            query_pos=query_pos_bt1,
            condition=condition,
        )

        # Decoder expects output_pos=[B, T, 1], output_val=[B, T, D]
        output_pos = input_pos_bt1
        output_val = input_feat_btd

        pred = self.decoder(
            x=latent,
            output_pos=output_pos,
            output_val=output_val,
            condition=condition,
        )
        # decoder already outputs [B, D, T] via its unbatch rearrange

        return pred
