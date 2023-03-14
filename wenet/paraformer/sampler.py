from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn

from wenet.paraformer.decoder import ParaformerDecoder
from wenet.utils.mask import make_pad_mask

class Sampler(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        embed_type: str = "attention",
        sampling_ratio: float = 0.2
    ):
        """
        Sampler after get encoder embedded acoustic features
        https://github.com/FLC777/GLAT
        """
        super().__init__()
        self.sampling_ratio = sampling_ratio
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(
        self,
        decoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        pred_acoustic_embeds: torch.Tensor,
        ignore_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        decoder 1st pass, but not with grads
        encoder_out: outputs of encoder (#Batch, time, dim)
        """
        batch_size, max_len = ys_pad.size()
        tgt_mask = ~make_pad_mask(ys_pad_lens, max_len)
        ys_pad_mask = ys_pad * tgt_mask
        ys_pad_embed = self.embed(ys_pad_mask)
        # predicted token sequence
        pred_tokens = decoder_out.argmax(-1)
        # find token should be discarded
        ignore = (ys_pad == ignore_id)
        # compute the number of pred equals to groundtruth
        same_num = (ys_pad == pred_tokens).masked_fill(ignore, 0).sum(1)
        # mask for getting embeddings from pred and target
        input_mask = torch.zeros_like(ignore)
        # number of choosing how many embeddings from target
        total_replace_nums = torch.tensor(0, device=ys_pad.device)
        for b in range(batch_size):
            target_num = ((ys_pad_lens[b] - same_num[b]) * self.sampling_ratio).long()
            if target_num > 0:
                input_mask[b].scatter_(
                    dim=0, 
                    index=torch.randperm(ys_pad_lens[b])[:target_num].to(decoder_out.device),
                    value=1
                )
                total_replace_nums += target_num
            
        total_same_num = same_num.sum()
        total_num = (~ignore).sum()
        sampler_info: Dict[str, Optional[torch.Tensor]] = {
            "total": total_num,
            "same": total_same_num,
            "replace": total_replace_nums
        }
        tgt_mask = tgt_mask.unsqueeze(2)
        input_mask = input_mask.eq(1)
        input_mask = input_mask.masked_fill(ignore, 0)
        input_mask_expand_dim = input_mask.unsqueeze(2).to(decoder_out.device)
        semantic_embeds = pred_acoustic_embeds.masked_fill(input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(~input_mask_expand_dim, 0)
        return semantic_embeds * tgt_mask, ys_pad_embed * tgt_mask, pred_acoustic_embeds * tgt_mask, tgt_mask, sampler_info