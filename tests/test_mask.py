import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.utils.mask import compute_valid_mask


def test_compute_valid_mask():
    state = [1, 0, True, False]
    mask = compute_valid_mask(state)

    expected = torch.tensor([True, False, True, False])
    assert torch.equal(mask, expected)
    assert mask.dtype == torch.bool
