"""Mask computation utilities."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def compute_valid_mask(state: Iterable[bool]) -> Tensor:
    """Compute a boolean mask indicating which actions are valid.

    The ``state`` argument is expected to be an iterable of truthy/falsy
    values, where each element corresponds to the validity of an action.
    The function returns a :class:`torch.Tensor` of dtype ``torch.bool``.
    """
    return torch.as_tensor(list(state), dtype=torch.bool)
