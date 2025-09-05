"""Train a Behavioral Cloning policy from processed trajectories.

This script expects a directory containing trajectory files saved as `.npz`.
Each file should contain two arrays: ``obs`` with shape ``(N, obs_dim)`` and
``acts`` with shape ``(N,)`` representing observations and discrete action
indices respectively.  The trained policy is saved under ``data/models``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # gymnasium is preferred, but fall back to gym if unavailable
    from gymnasium import spaces
except ImportError:  # pragma: no cover - depends on environment
    from gym import spaces

from imitation.algorithms import bc
from imitation.data import types


def load_transitions(data_dir: Path) -> types.Transitions:
    """Load all trajectory files in ``data_dir`` and return a ``Transitions``.

    Parameters
    ----------
    data_dir:
        Directory containing ``.npz`` files with ``obs`` and ``acts`` arrays.

    Returns
    -------
    types.Transitions
        Concatenated transitions from all trajectory files found.
    """
    obs_list: list[np.ndarray] = []
    acts_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    dones_list: list[np.ndarray] = []

    for file in sorted(data_dir.glob("*.npz")):
        data = np.load(file)
        obs = data["obs"]
        acts = data["acts"]
        if len(obs) != len(acts):
            raise ValueError(
                f"In {file}, obs and acts have mismatched lengths: {len(obs)} != {len(acts)}"
            )
        next_obs = data.get("next_obs")
        if next_obs is None:
            # Placeholder next observations; not required for BC but part of API
            next_obs = np.zeros_like(obs)
        dones = data.get("dones")
        if dones is None:
            dones = np.zeros(len(acts), dtype=bool)
        obs_list.append(obs)
        acts_list.append(acts)
        next_obs_list.append(next_obs)
        dones_list.append(dones)

    if not obs_list:
        raise FileNotFoundError(f"No trajectory files found in {data_dir!s}")

    return types.Transitions(
        obs=np.concatenate(obs_list),
        acts=np.concatenate(acts_list),
        next_obs=np.concatenate(next_obs_list),
        dones=np.concatenate(dones_list),
        infos=None,
    )


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for training a BC policy."""
    parser = argparse.ArgumentParser(description="Train BC policy from trajectories")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing processed trajectory .npz files",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/bc_policy.pt"),
        help="Output path for the trained model",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    transitions = load_transitions(args.data_dir)
    obs_dim = transitions.obs.shape[1]
    n_actions = int(transitions.acts.max()) + 1

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Discrete(n_actions)

    trainer = bc.BC(observation_space, action_space, demonstrations=transitions)
    trainer.train(n_epochs=args.epochs)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.policy.save(args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
