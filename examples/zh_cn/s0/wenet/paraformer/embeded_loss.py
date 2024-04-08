import torch
import torch.nn as nn

class EmbedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_layer = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    def forward(
        self,
        pred_acoustic_embeds: torch.Tensor,
        ys_pad_embed: torch.Tensor,
        tgt_mask: torch.Tensor
    ):
        tgt_mask = tgt_mask.squeeze(2)
        batch_size = pred_acoustic_embeds.size(0)
        cos_loss = 1 - self.cos_layer(ys_pad_embed, pred_acoustic_embeds)
        cos_loss = cos_loss * tgt_mask
        cos_loss = torch.sum(cos_loss)
        cos_loss = cos_loss / batch_size
        return cos_loss

    