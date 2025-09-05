"""Model definitions for the AoE2DE agent.

This module currently exposes :class:`SimpleTransformer`, a minimal
Transformer-based policy network used by the training and inference
scripts.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleTransformer(nn.Module):
    """Minimal Transformer-based policy network for AoE2DE.

    Parameters
    ----------
    state_dim:
        Dimensionality of the input state representation.
    num_actions:
        Number of discrete actions the agent can take.
    d_model:
        Size of the model's hidden representations. Defaults to ``64``.
    nhead:
        Number of attention heads in each Transformer encoder layer.
        Defaults to ``4``.
    num_layers:
        Number of Transformer encoder layers. Defaults to ``2``.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.policy = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action logits for a batch of states.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len, state_dim)`` representing a
            batch of state sequences.

        Returns
        -------
        torch.Tensor
            Logits over actions of shape ``(batch, num_actions)`` corresponding
            to the final state in each sequence.
        """

        x = self.input_proj(x)
        x = self.encoder(x)
        return self.policy(x[:, -1])
