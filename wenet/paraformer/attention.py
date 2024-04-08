import math

import torch
import torch.nn as nn

from typing import Optional, Tuple

class MultiHeadedAttentionSANM(nn.Module):
    def __init__(self, n_head: int, n_feat: int, kernel_size: int, dropout_rate: float, sanm_shift: int = 0):
        """Construct a san-m attention objetc"""
        # in funasr, there are two feat dim, input and output
        # but in the large model, they are the same. so there
        # we will not set two parameters.
        super(MultiHeadedAttentionSANM, self).__init__()
        assert n_feat % n_head == 0
        # we assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # To keep the same attention arch within wenet, we will
        # still use three nn.Linear module to do generate the
        # q, k and v while they are merge into one module in funasr.
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # fsmn block
        # this is a depwise convolutional module.
        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        # to keep the size of the output of fsmn same with the input
        # we need to do the padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        
    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v
        
    def forward_fsmn(self, x: torch.Tensor,
                     mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                     mask_shift_chunk: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Do the FSMN.
        x: batch, head, t, dk
        so, fsmn is a added module for value through a conv1d module.
        """
        n_batch = x.size(0)
        x = x.transpose(1, 2)
        x = x.view(n_batch, -1, self.d_k * self.h)
        
        mask = mask.view(n_batch, -1, 1)
        if mask_shift_chunk is not None:
            mask = mask * mask_shift_chunk
        # x = x * mask
        residual = x
        x = x.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)
        
        # return x * mask
        return x
        
    def forward_attention(self, 
                          value: torch.Tensor, scores: torch.Tensor, 
                          mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                          mask_att_chunk_encoder: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0 :  # time2 > 0
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)
        
        
    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                mask_shift_chunk: Optional[torch.Tensor] = None,
                mask_att_chunk_encoder: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        q, k, v = self.forward_qkv(query, key, value)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shift_chunk)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        att_outs = self.forward_attention(v, scores, mask, mask_att_chunk_encoder)
        new_cache = torch.ones((0, 0, 0, 0))
        return att_outs + fsmn_memory, new_cache
        
        
        
        
