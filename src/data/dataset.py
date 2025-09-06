"""PyTorch dataset for AoE2DE behavioral cloning episodes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    """Dataset reading episodes stored as JSON Lines files.

    Each line in the ``.jsonl`` file is expected to contain three keys:

    ``state``
        Mapping of feature names to numeric values.
    ``action_id``
        Integer identifier of the action taken in that state.
    ``valid_action_mask``
        Sequence indicating which actions were legal.  Elements are
        interpreted as ``0`` or ``1``.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file containing the episodes.
    split:
        Optional split selector.  Pass ``"train"`` or ``"val"`` to select a
        subset of the data.  When ``None`` (default) the entire dataset is
        used.
    val_ratio:
        Fraction of episodes reserved for the validation split.  Only used
        when ``split`` is provided.
    seed:
        Random seed controlling the train/validation shuffling.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: Optional[str] = None,
        val_ratio: float = 0.1,
        seed: int = 0,
    ) -> None:
        self._episodes: List[Dict[str, object]] = []
        self._load(path)

        if split is not None:
            if split not in {"train", "val"}:
                raise ValueError("split must be either 'train', 'val' or None")
            self._apply_split(split, val_ratio, seed)

    # ------------------------------------------------------------------
    def _load(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"episode file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self._episodes.append(json.loads(line))

    # ------------------------------------------------------------------
    def _apply_split(self, split: str, val_ratio: float, seed: int) -> None:
        total = len(self._episodes)
        indices = list(range(total))
        rng = random.Random(seed)
        rng.shuffle(indices)
        val_size = int(total * val_ratio)
        if split == "val":
            selected = indices[:val_size]
        else:  # train
            selected = indices[val_size:]
        self._episodes = [self._episodes[i] for i in selected]

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._episodes)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode = self._episodes[index]

        state_dict: Dict[str, float] = episode["state"]  # type: ignore[assignment]
        # Convert the state dictionary into a stable ordered tensor by sorting
        # keys alphabetically.  This keeps feature ordering deterministic
        # regardless of how the data was serialized.
        state_vals: List[float] = [state_dict[k] for k in sorted(state_dict)]
        state = torch.tensor(state_vals, dtype=torch.float32)

        action_id = torch.tensor(episode["action_id"], dtype=torch.long)

        mask_seq: Sequence[int] = episode.get("valid_action_mask", [])  # type: ignore[assignment]
        valid_action_mask = torch.tensor(mask_seq, dtype=torch.float32)

        return {
            "state": state,
            "action_id": action_id,
            "valid_action_mask": valid_action_mask,
        }
