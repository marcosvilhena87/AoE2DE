"""Utilities to train the imitation-learning agent.

This script loads a configuration file that specifies hyperparameters,
constructs a dataset from preprocessed episodes and trains a simple model
using supervised imitation learning. The resulting weights and training
metrics are saved to an output directory, and TensorBoard support is
available for optional visualisation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard not installed
    SummaryWriter = None  # type: ignore

from src.agent.model import SimpleTransformer


@dataclass
class TrainConfig:
    """Configuration for training with basic validation."""

    data_dir: Path
    state_dim: int
    num_actions: int
    batch_size: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    optimizer: str = "Adam"
    lr: float = 1e-3
    tensorboard: bool = True
    val_split: float = 0.2
    patience: int = 5

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.exists():
            raise ValueError("data_dir does not exist")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        for name in ["batch_size", "d_model", "nhead", "num_layers", "patience"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 < self.val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        if not hasattr(optim, self.optimizer):
            raise ValueError(f"Unknown optimizer: {self.optimizer}")


def load_config(path: Path) -> TrainConfig:
    """Load and validate a YAML or JSON configuration file."""

    if path.suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return TrainConfig(**data)


class EpisodeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of state/action pairs extracted from preprocessed episodes."""

    def __init__(self, data_dir: Path) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for file in sorted(data_dir.glob("*.pt")):
            episode = torch.load(file)
            states = episode.get("states") or episode.get("obs")
            actions = episode.get("actions")
            if states is None or actions is None:
                continue
            for s, a in zip(states, actions):
                self.samples.append((s, a))

    def __len__(self) -> int:  # pragma: no cover - simple container
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_agent(config_path: str | Path, output_dir: str | Path, epochs: int,
                use_gpu: bool, seed: int | None = None) -> None:
    """Train the imitation agent using the provided configuration file."""

    cfg = load_config(Path(config_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    dataset = EpisodeDataset(cfg.data_dir)
    train_size = int(len(dataset) * (1 - cfg.val_split))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed or 0),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_size > 0 else None

    model = SimpleTransformer(state_dim=cfg.state_dim,
                              num_actions=cfg.num_actions,
                              d_model=cfg.d_model,
                              nhead=cfg.nhead,
                              num_layers=cfg.num_layers)
    model.to(device)

    optim_cls = getattr(optim, cfg.optimizer)
    optimizer = optim_cls(model.parameters(), lr=cfg.lr)

    ckpt_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best_model.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    writer = (
        SummaryWriter(log_dir=str(output_dir / "tb"))
        if (cfg.tensorboard and SummaryWriter is not None)
        else None
    )

    best_loss = float("inf")
    patience_count = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * states.size(0)

        train_loss = total_loss / max(len(train_ds), 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for states, actions in val_loader:
                    states, actions = states.to(device), actions.to(device)
                    logits = model(states)
                    loss = F.cross_entropy(logits, actions)
                    val_total += loss.item() * states.size(0)
            val_loss = val_total / max(len(val_ds), 1)

        history.append({"epoch": float(epoch), "train_loss": train_loss,
                         "val_loss": float(val_loss) if val_loss is not None else None})
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss is not None:
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict()}, best_path)
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    break

    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, ckpt_path)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    if writer is not None:
        writer.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AoE2DE agent.")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to store outputs")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser


def main() -> None:  # pragma: no cover - entry point
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_agent(args.config, args.output_dir, args.epochs, args.use_gpu, args.seed)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

