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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file.

    Parameters
    ----------
    path:
        Location of the configuration file.
    """

    if path.suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise ValueError(f"Unsupported config format: {path.suffix}")


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


class SimpleTransformer(nn.Module):
    """Minimal Transformer-based policy network."""

    def __init__(self, state_dim: int, num_actions: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.policy = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple passthrough
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.policy(x[:, -1])


def train_agent(config_path: str | Path, output_dir: str | Path, epochs: int,
                use_gpu: bool) -> None:
    """Train the imitation agent using the provided configuration file."""

    cfg = load_config(Path(config_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    data_dir = Path(cfg["data_dir"])  # directory containing preprocessed episodes
    dataset = EpisodeDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.get("batch_size", 32),
                            shuffle=True)

    model = SimpleTransformer(state_dim=cfg["state_dim"],
                              num_actions=cfg["num_actions"],
                              d_model=cfg.get("d_model", 64),
                              nhead=cfg.get("nhead", 4),
                              num_layers=cfg.get("num_layers", 2))
    model.to(device)

    optim_cls = getattr(optim, cfg.get("optimizer", "Adam"))
    optimizer = optim_cls(model.parameters(), lr=cfg.get("lr", 1e-3))

    ckpt_path = output_dir / "checkpoint.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if cfg.get(
        "tensorboard", True) else None

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * states.size(0)

        avg_loss = total_loss / max(len(dataset), 1)
        history.append({"epoch": float(epoch), "loss": avg_loss})
        if writer is not None:
            writer.add_scalar("loss/train", avg_loss, epoch)

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
    return parser


def main() -> None:  # pragma: no cover - entry point
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_agent(args.config, args.output_dir, args.epochs, args.use_gpu)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

