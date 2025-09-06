import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.policy import MLPPolicy, TransformerPolicy


def _check_mask_application(model, state: torch.Tensor, mask: torch.Tensor) -> None:
    # Training mode
    model.train()
    logits = model(state, mask)
    assert torch.isinf(logits[~mask]).all()

    # Inference mode
    model.eval()
    logits = model(state, mask)
    assert torch.isinf(logits[~mask]).all()


def test_mlp_policy_masking():
    model = MLPPolicy(input_dim=4, hidden_dim=8, output_dim=3)
    state = torch.randn(2, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    _check_mask_application(model, state, mask)


def test_transformer_policy_masking():
    model = TransformerPolicy(input_dim=6, num_heads=2, num_layers=1, output_dim=3)
    state = torch.randn(2, 5, 6)
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    _check_mask_application(model, state, mask)
