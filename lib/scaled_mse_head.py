from typing import Dict, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from lib.action_head import fan_in_linear
from lib.normalize_ewma import NormalizeEwma


class ScaledMSEHead(nn.Module):
    """
    Linear output layer that scales itself so that targets are always normalized to N(0, 1)
    """

    def __init__(
        self, input_size: int, output_size: int, norm_type: Optional[str] = "ewma", norm_kwargs: Optional[Dict] = None,
        n_heads: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.norm_type = norm_type
        self.n_heads = n_heads

        self.linear = nn.ModuleList([
            nn.Linear(self.input_size, self.output_size)
            for _ in range(n_heads)
        ])

        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.normalizer = nn.ModuleList([
            NormalizeEwma(output_size, **norm_kwargs)
            for _ in range(n_heads)
        ])

    def reset_parameters(self):
        for i in range(self.n_heads):
            init.orthogonal_(self.linear[i].weight)
            fan_in_linear(self.linear[i])
            self.normalizer[i].reset_parameters()

    def forward(self, input_data, head_id=None):
        if head_id is None:
            return self.linear[0](input_data)

        assert len(head_id) == input_data.shape[0]
        out = []
        for i in range(len(head_id)):
            assert head_id[i] >= 0 and head_id[i] < len(self.linear), \
                "Found out of range head id {}. Expected range [0, {}).".format(head_id, len(self.linear))
            out.append(self.linear[head_id[i]](input_data[i:i+1]))

        return th.cat(out, dim=0)

    def loss(self, prediction, target, head_id=None):
        """
        Calculate the MSE loss between output and a target.
        'Prediction' has to be normalized while target is denormalized.
        Loss is calculated in a 'normalized' space.
        """
        if head_id is None:
            return F.mse_loss(prediction, self.normalizer[0](target), reduction="mean")

        assert len(head_id) == target.shape[0]
        out = []
        for i in range(len(head_id)):
            assert head_id[i] >= 0 and head_id[i] < len(self.normalizer), \
                "Found out of range head id {}. Expected range [0, {}).".format(head_id, len(self.normalizer))
            out.append(F.mse_loss(prediction, self.normalizer[head_id[i]](target[i:i+1]), reduction="mean"))
            
        return th.cat(out, dim=0)

    def denormalize(self, input_data, head_id=None):
        """Convert input value from a normalized space into the original one"""
        if head_id is None:
            return self.normalizer[0].denormalize(input_data)

        assert len(head_id) == input_data.shape[0]
        out = []
        for i in range(len(head_id)):
            assert head_id[i] >= 0 and head_id[i] < len(self.normalizer), \
                "Found out of range head id {}. Expected range [0, {}).".format(head_id, len(self.normalizer))
            out.append(self.normalizer[head_id[i]].denormalize(input_data[i:i+1]))
            
        return th.cat(out, dim=0)

    def normalize(self, input_data, head_id=None):
        if head_id is None:
            return self.normalizer[0](input_data)

        assert len(head_id) == input_data.shape[0]
        out = []
        for i in range(len(head_id)):
            assert head_id[i] >= 0 and head_id[i] < len(self.normalizer), \
                "Found out of range head id {}. Expected range [0, {}).".format(head_id, len(self.normalizer))
            out.append(self.normalizer[head_id[i]](input_data[i:i+1]))
            
        return th.cat(out, dim=0)
