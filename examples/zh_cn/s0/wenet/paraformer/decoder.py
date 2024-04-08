# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Decoder definition."""
from typing import Tuple, List, Optional

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.paraformer.decoder_layer import DecoderLayerSANM
from wenet.paraformer.fsmn import FSMN
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForwardSANM
from wenet.utils.mask import (subsequent_mask, make_pad_mask)

class ParaformerSANMDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        kernel_size: int = 11,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayerSANM(
                attention_dim,
                FSMN(attention_dim, self_attention_dropout_rate, kernel_size),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForwardSANM(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])
        self.ff_layer = PositionwiseFeedForwardSANM(attention_dim, linear_units, dropout_rate)
        self.ff_norm = torch.nn.LayerNorm(attention_dim)


    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        x = ys_in_pad
        maxlen = x.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(x.device)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask, _ = layer(x, tgt_mask, memory,
                                                     memory_mask)
        
        if self.normalize_before:
            x = self.ff_norm(x)
            
        x = self.ff_layer(x)
        
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens


class ParaformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        x = ys_in_pad
        maxlen = x.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(x.device)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens


class BiParaformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):

        assert check_argument_types()
        super().__init__()
        self.left_decoder = ParaformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            use_output_layer, normalize_before, concat_after)

        self.right_decoder = ParaformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            use_output_layer, normalize_before, concat_after)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens
