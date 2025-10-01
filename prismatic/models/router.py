import logging
from typing import Literal

import einops
import torch
import torch.nn as nn

from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConfig, DeformableDetrDecoderLayer
)

class MoEAggregator(nn.Module):
    """
    MoEAggregator is a module that aggregates the outputs of the experts

    num_experts (int): Number of experts
    seq_dim (int): The dimension of the input sequence
    router_method (str): The method to route the input sequence to the experts
    """

    def __init__(
            self,
            num_experts=2,
            seq_dim=256,
            router_method: Literal["mlp", "add"] = "mlp"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router_method = router_method
        self.routing_outcome = None
        if router_method == 'mlp':
            self.router = nn.Sequential(
                nn.Linear(seq_dim, seq_dim),
                nn.GELU(),
                nn.Linear(seq_dim, num_experts),
            )
        else:
            assert router_method == "add", f"router_method({router_method}) not supported"

    def forward(self, inputs_embeds, seq_embeds):
        """
        inputs_embeds (List of `torch.FloatTensor` of shape `(batch_size, length, hidden_size)`)
        """
        hidden_states = torch.zeros_like(inputs_embeds[0])
        bs = inputs_embeds[0].shape[0]
        self.routing_outcome = []

        if self.router_method == 'add':
            ratios = torch.ones(bs, len(inputs_embeds), device=inputs_embeds[0].device)
        else:
            ratios = self.router(seq_embeds)
            ratios = torch.softmax(ratios, dim=-1)

        for i in range(self.num_experts):
            _weighted_embeds = ratios[:, i].view(-1, 1, 1) * inputs_embeds[i]
            hidden_states += _weighted_embeds
            self.routing_outcome.append(_weighted_embeds.abs().mean(dim=-1).mean(dim=-1))

        self.routing_outcome = torch.stack(self.routing_outcome, dim=1)
        return hidden_states