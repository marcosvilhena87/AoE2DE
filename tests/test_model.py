from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from run_agent import load_model
from src.agent.model import SimpleTransformer


def test_forward_pass() -> None:
    model = SimpleTransformer(state_dim=4, num_actions=3, d_model=8, nhead=2, num_layers=1)
    x = torch.randn(2, 5, 4)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_weight_loading(tmp_path: Path) -> None:
    model = SimpleTransformer(state_dim=4, num_actions=3, d_model=8, nhead=2, num_layers=1)
    ckpt = {"model": model.state_dict()}
    weights = tmp_path / "weights.pt"
    torch.save(ckpt, weights)

    loaded = load_model(weights, torch.device("cpu"))
    for p_orig, p_loaded in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p_orig, p_loaded)
