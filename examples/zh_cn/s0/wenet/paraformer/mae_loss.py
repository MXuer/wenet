import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(
        self, 
        normalize_length: bool = False
        ):
        super(MAELoss, self).__init__()
        self.normalize_length = normalize_length
        # none means get loss's mean value
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, token_length, pred_token_length):
        batch = token_length.size(0)
        total = token_length.sum()
        token_length = token_length.float()
        loss = self.criterion(token_length, pred_token_length)
        denom = total if self.normalize_length else total
        return loss / denom
