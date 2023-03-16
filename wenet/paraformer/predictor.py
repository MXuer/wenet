from typing import Tuple, Optional

import torch
import torch.nn as nn
from typeguard import check_argument_types

from wenet.transformer.convolution import ConvolutionModule
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask


class ConvPredictor(nn.Module):
    def __init__(
        self, 
        input_size: int,
        l_order: int = 1,
        r_order: int = 1,
    ):
        super(ConvPredictor, self).__init__()
        self.pad = nn.ConstantPad1d((l_order, r_order), 0.)
        self.conv1d = nn.Conv1d(input_size, input_size, l_order + r_order + 1)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pad(x)
        x = self.conv1d(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        return x

class Predictor(nn.Module):
    def __init__(
        self,
        input_size,
        use_dynamic_chunk: bool = True,
        use_dynamic_left_chunk: bool = False,
        static_chunk_size: int = 0,
        casual: bool = True,
        cnn_module_kernel: int = 15,
        cnn_module_norm: str = "batch_norm",
        activation_type: str = "swish",
        smooth_factor: float = 1.0,
        noise_threshold: float = 0.0,
        tail_threshold: float = 0.5,
        fire_threshold: float = 1.0,
        ignore_id: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        activation = get_activation(activation_type)
        
        #-> START
        # # Original
        # self.convolution_layer = ConvPredictor(input_size)
        
        # Wenet Internal
        convolution_layer_args = (input_size, cnn_module_kernel, activation, cnn_module_norm, casual)
        self.convolution_layer = ConvolutionModule(*convolution_layer_args)
        #-> END
        
        self.linear_layer = nn.Linear(input_size, 1)
        
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.static_chunk_size = static_chunk_size
        self.fire_threshold = fire_threshold
        self.tail_threshold = tail_threshold
        self.ignore_id = ignore_id
        
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0)),
        ):
        """
        compute output for predictor
        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask Tensor for the input (#batch, time, time)
        """
        # (B, T)
        T = x.size(1)
        # (B, 1, T)
        masks = ~make_pad_mask(x_lens, T).unsqueeze(1)
        # (B, T, F)
        hidden, *_ = self.convolution_layer(x)
        # (B, T, 1)
        hidden = self.linear_layer(hidden)
        alphas = torch.sigmoid(hidden)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        masks = masks.transpose(1, -1)
        alphas = alphas * masks
        alphas = alphas.squeeze(-1)
        predicted_token_nums = alphas.sum(-1)
        if target is not None:
            target_token_num = (target != self.ignore_id).sum(-1)
            alphas *= (target_token_num / predicted_token_nums).unsqueeze(1)
        fired_frames, fires, r_fired_frames = self.cif(x, alphas)
        return fired_frames, predicted_token_nums, r_fired_frames
        
    def cif(self,
            x: torch.Tensor,
            alphas: torch.Tensor,
    ):
        """
        compute the continous intergrated-fire
        """
        batch_size, len_time, hidden_size = x.size()
        
        intergrate = torch.zeros([batch_size], device=x.device)
        frame = torch.zeros([batch_size, hidden_size], device=x.device)
        
        fires = torch.zeros([batch_size, len_time], device=x.device)
        frames = torch.zeros([batch_size, len_time, hidden_size], device=x.device)
        
        thresholds = torch.full([batch_size,], self.fire_threshold, device=x.device)
        
        # each time step get a scaled frame. 
        # if fired: the frame for next step will be (beta - prev_alpha) * current_frame
        # if not fired: the frame for next step will be prev_frame + alpha * current_frame
        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = thresholds - intergrate
            intergrate += alpha
            fires[:, t] = intergrate
            # fired or not for batch, (batch)
            fire_place = intergrate >= self.fire_threshold
            # fire: intial intergrate for next time is intergrate - fire_threshold
            # not fire: keep same with current intergrate
            intergrate = torch.where(fire_place,
                                     intergrate - thresholds,
                                     intergrate)
            # fire: remains for current frame is fire_threshold - last frame's intergrate
            # not fire: remains for current frame is alpha
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            # current frame remains
            # fire: remains for next fire self.fire_threshold - last intergrated
            # not fire: 0 (not trigger current fire position)
            remains = alpha - cur
            # add aplha scaled current value to frame
            frame += cur.unsqueeze(1) * x[:, t, :]
            frames[:, t, :] = frame
            # fire: change the fired postion to 1-alpha scale value
            # not fire: keep same
            frame = torch.where(fire_place.unsqueeze(1),
                                remains.unsqueeze(1) * x[:, t, :],
                                frame)


        # num of predicted labels
        num_predict_labels = torch.round(alphas.sum(-1)).int()
        # the max number predicted labels in a batch
        max_num_labels_batch = int(num_predict_labels.max().item())
        fired_frames_batch = torch.zeros([batch_size, max_num_labels_batch, hidden_size], device=x.device)
        # for reverse decoder's input
        r_fired_frames_batch = torch.zeros([batch_size, max_num_labels_batch, hidden_size], device=x.device)
        for b in range(batch_size):
            fire = fires[b, :]
            # find all the frames which were fired
            fired_frames = torch.index_select(frames[b, :, :], 
                                   0, 
                                   torch.nonzero(fire >= self.fire_threshold).squeeze(1))
            fired_frames_batch[b, :fired_frames.size(0), :] = fired_frames

            # reverse in the time axis
            r_fired_frames = torch.flip(fired_frames, dims=[1])
            fired_frames_batch[b, :fired_frames.size(0), :] = fired_frames
            r_fired_frames_batch[b, :fired_frames.size(0), :] = r_fired_frames
            
        return fired_frames_batch, fires, r_fired_frames_batch

    
class PredictorSeperable(nn.Module):
    def __init__(
        self,
        input_size,
        use_dynamic_chunk: bool = True,
        use_dynamic_left_chunk: bool = False,
        static_chunk_size: int = 0,
        casual: bool = True,
        cnn_module_kernel: int = 15,
        cnn_module_norm: str = "batch_norm",
        activation_type: str = "swish",
        smooth_factor: float = 1.0,
        noise_threshold: float = 0.0,
        tail_threshold: float = 0.5,
        fire_threshold: float = 1.0,
        ignore_id: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        activation = get_activation(activation_type)
        convolution_layer_args = (input_size, cnn_module_kernel, activation, cnn_module_norm, casual)
        self.convolution_layer = ConvolutionModule(*convolution_layer_args)
        self.linear_layer = nn.Linear(input_size, 1)
        
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.static_chunk_size = static_chunk_size
        self.fire_threshold = fire_threshold
        self.tail_threshold = tail_threshold
        self.ignore_id = ignore_id
        
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0)),
        ):
        """
        compute output for predictor
        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask Tensor for the input (#batch, time, time)
        """
        # (B, T)
        T = x.size(1)
        # (B, 1, T)
        masks = ~make_pad_mask(x_lens, T).unsqueeze(1)
        # (B, T, F)
        hidden, new_cnn_cache = self.convolution_layer(x, masks, cnn_cache)
        # (B, T, 1)
        hidden = self.linear_layer(hidden)
        alphas = torch.sigmoid(hidden)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        masks = masks.transpose(1, -1)
        alphas = alphas * masks
        alphas = alphas.squeeze(-1)
        predicted_token_nums = alphas.sum(-1)
        if target is not None:
            target_token_num = (target != self.ignore_id).sum(-1)
            alphas *= (target_token_num / predicted_token_nums).unsqueeze(1)
        fired_frames, fires, r_fired_frames = self.cif(x, alphas)
        return fired_frames, predicted_token_nums, r_fired_frames
        
    def cif(self,
            x: torch.Tensor,
            alphas: torch.Tensor,
    ):
        """
        compute the continous intergrated-fire
        """
        batch_size, len_time, hidden_size = x.size()
        
        intergrate = torch.zeros([batch_size], device=x.device)
        frame = torch.zeros([batch_size, hidden_size], device=x.device)
        
        fires = torch.zeros([batch_size, len_time], device=x.device)
        frames = torch.zeros([batch_size, len_time, hidden_size], device=x.device)
        
        thresholds = torch.full([batch_size,], self.fire_threshold, device=x.device)
        
        # each time step get a scaled frame. 
        # if fired: the frame for next step will be (beta - prev_alpha) * current_frame
        # if not fired: the frame for next step will be prev_frame + alpha * current_frame
        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = thresholds - intergrate
            intergrate += alpha
            fires[:, t] = intergrate
            # fired or not for batch, (batch)
            fire_place = intergrate >= self.fire_threshold
            # fire: intial intergrate for next time is intergrate - fire_threshold
            # not fire: keep same with current intergrate
            intergrate = torch.where(fire_place,
                                     intergrate - thresholds,
                                     intergrate)
            # fire: remains for current frame is fire_threshold - last frame's intergrate
            # not fire: remains for current frame is alpha
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            # current frame remains
            # fire: remains for next fire self.fire_threshold - last intergrated
            # not fire: 0 (not trigger current fire position)
            remains = alpha - cur
            # add aplha scaled current value to frame
            frame += cur.unsqueeze(1) * x[:, t, :]
            frames[:, t, :] = frame
            # fire: change the fired postion to 1-alpha scale value
            # not fire: keep same
            frame = torch.where(fire_place.unsqueeze(1),
                                remains.unsqueeze(1) * x[:, t, :],
                                frame)


        # num of predicted labels
        num_predict_labels = torch.round(alphas.sum(-1)).int()
        # the max number predicted labels in a batch
        max_num_labels_batch = int(num_predict_labels.max().item())
        fired_frames_batch = torch.zeros([batch_size, max_num_labels_batch, hidden_size], device=x.device)
        # for reverse decoder's input
        r_fired_frames_batch = torch.zeros([batch_size, max_num_labels_batch, hidden_size], device=x.device)
        for b in range(batch_size):
            fire = fires[b, :]
            # find all the frames which were fired
            fired_frames = torch.index_select(frames[b, :, :], 
                                   0, 
                                   torch.nonzero(fire >= self.fire_threshold).squeeze(1))
            fired_frames_batch[b, :fired_frames.size(0), :] = fired_frames

            # reverse in the time axis
            r_fired_frames = torch.flip(fired_frames, dims=[1])
            fired_frames_batch[b, :fired_frames.size(0), :] = fired_frames
            r_fired_frames_batch[b, :fired_frames.size(0), :] = r_fired_frames
            
        return fired_frames_batch, fires, r_fired_frames_batch


