from typing import Tuple, Optional

import torch
import torch.nn as nn

class FSMN(nn.Module):
    """
    FSMN block
    """

    def __init__(
        self, 
        n_feat: int, 
        dropout_rate:float, 
        kernel_size: int, 
        sanm_shfit: int = 0
    ):
        """Construct an MultiHeadedAttention object."""
        super(FSMN, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(n_feat, n_feat,
                                    kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.kernel_size = kernel_size

    def forward(
        self, 
        inputs: torch.Tensor, 
        mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None, 
        mask_shfit_chunk: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        :param x: (#batch, time1, size).
        :param mask: Mask tensor (#batch, 1, time)
        :return:
        '''
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b ,-1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        b, d, t = x.size()
        if cache is None:
            x = self.pad_fn(x)
            cache = x
        else:
            x = torch.cat((cache[:, :, 1:], x), dim=2)
            x = x[:, :, -self.kernel_size:]
            cache = x
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]

        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x, cache