"""Train a campaign-playing agent from preprocessed data."""

from __future__ import annotations

import argparse
from pathlib import Path


def train_agent(config_path: str | Path) -> None:
    """Train the imitation agent using the provided configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML/JSON configuration file.
    """
    cfg = Path(config_path)
    # Placeholder implementation – real training logic would be invoked here.
    print(f"Pretending to train agent with config at {cfg}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AoE2DE agent.")
    parser.add_argument("--config", required=True, help="Path to config file")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_agent(args.config)


if __name__ == "__main__":
    main()
