# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Optional

class GlobalCMVN(torch.nn.Module):
    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True,
                 lfr_m: int = 1,
                 lfr_n: int = 1):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)


    def apply_lfr(self, inputs: torch.Tensor) -> torch.Tensor:
        LFR_inputs = []
        B, T, F = inputs.size()
        T_lfr = int(torch.ceil(torch.tensor(T / self.lfr_n)))
        left_padding = inputs[0].repeat((self.lfr_m - 1) // 2, 1)
        left_padding = inputs[:, :1, :].repeat(1, (self.lfr_m - 1) // 2, 1)
        inputs = torch.cat((left_padding, inputs), dim=1)
        T = T + (self.lfr_m - 1) // 2
        for i in range(T_lfr):
            if self.lfr_m <= T - i * self.lfr_n:
                LFR_inputs.append((inputs[:, i * self.lfr_n:i * self.lfr_n + self.lfr_m]).view(B, 1, -1))
            else:  # process last LFR frame
                num_padding = self.lfr_m - (T - i * self.lfr_n)
                frame = (inputs[:, i * self.lfr_n:]).view(-1)
                for _ in range(num_padding):
                    frame = torch.hstack((frame, inputs[:, -1]))
                LFR_inputs.append(frame)
        LFR_outputs = torch.stack(LFR_inputs, dim=1).squeeze()
        return LFR_outputs.type(torch.float32)

    def forward(self, x: torch.Tensor, xs_lens: torch.Tensor = torch.ones((0, 0))):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        if self.lfr_m != 1 or self.lfr_n != 1:
            x = self.apply_lfr(x)
            xs_lens = torch.ceil(torch.div(xs_lens, self.lfr_n)).long()
        return x, xs_lens
