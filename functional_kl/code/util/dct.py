import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_dct


class LinearDCT(nn.Module):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, dct_type=None, 
                 norm='ortho'):
        super().__init__()
        assert dct_type is not None
        self.dct_type = dct_type
        self.norm = norm
        self.register_buffer('weight', None, persistent=False)

    def initialize(self, height):
        # initialise using dct function
        I = torch.eye(height)
        if self.dct_type == 'dct':
            self.weight = torch_dct.dct(I, norm=self.norm).data.t()
        elif self.dct_type == 'idct':
            self.weight = torch_dct.idct(I, norm=self.norm).data.t()
        else:
            raise ValueError

    def forward(self, x):
        """Can be used with a LinearDCT layer to do a 2D DCT.
        :param x: the input signal
        :param linear_layer: any PyTorch Linear layer
        :return: result of linear layer applied to last 2 dimensions
        """

        if self.weight is None or self.weight.shape[-1] != x.shape[-1]:
            self.initialize(x.shape[-1])
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)

        X1 = F.linear(x, self.weight, None)
        return X1
 

    def extra_repr(self) -> str:
        return (f'(dct_type): {self.dct_type}\n'
                f'(norm): {self.norm}'
                )


class LinearDCT1d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=1, dct_type='dct', norm=norm)


class LinearIDCT1d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=1, dct_type='idct', norm=norm)

