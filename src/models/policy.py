"""Policy network implementations."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class MLPPolicy(nn.Module):
    """Simple multi-layer perceptron policy network.

    Parameters
    ----------
    input_dim:
        Size of the input state vector.
    hidden_dim:
        Number of units in the hidden layer.
    output_dim:
        Number of possible actions (size of the output logits).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute masked action logits.

        The ``mask`` should be a boolean tensor where ``False`` entries mark
        invalid actions. These entries are set to ``-inf`` in the returned
        logits, ensuring the mask is respected during both training and
        inference.
        """
        logits = self.net(state)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class TransformerPolicy(nn.Module):
    """Policy network based on a Transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute masked action logits using a Transformer encoder."""
        x = self.encoder(state)
        # Aggregate sequence dimension via mean pooling
        logits = self.fc(x.mean(dim=1))
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits
