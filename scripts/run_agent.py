"""Run a trained agent on an Age of Empires II: DE mission."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import torch

from train_agent import SimpleTransformer
from src.env import AoE2DEEnvironment


def load_model(weights: Path, device: torch.device) -> SimpleTransformer:
    """Load the policy network with weights from ``weights``."""

    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint["model"]

    # Infer architecture parameters from the state dict.
    state_dim = state_dict["input_proj.weight"].shape[1]
    d_model = state_dict["input_proj.weight"].shape[0]
    num_actions = state_dict["policy.weight"].shape[0]
    layer_keys = [k for k in state_dict if k.startswith("encoder.layers")]
    num_layers = max(int(k.split(".")[2]) for k in layer_keys) + 1 if layer_keys else 1

    model = SimpleTransformer(state_dim=state_dim, num_actions=num_actions,
                              d_model=d_model, num_layers=num_layers)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_agent(mission_name: str, weights: Path, time_limit: int,
              log_file: Path | None) -> None:
    """Execute the trained agent for a given campaign mission."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    env = AoE2DEEnvironment()

    history: List[dict] = []
    start = time.monotonic()
    while time.monotonic() - start < time_limit:
        state = env.read_state()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(state_tensor)
            action = int(torch.argmax(logits, dim=-1).item())
        env.send_action(action)
        logging.info("state=%s action=%s", state, action)
        history.append({"state": state, "action": action})
        if env.objectives_completed():
            logging.info("Mission objectives completed, shutting down.")
            break
        time.sleep(0.1)
    else:
        logging.info("Time limit of %s seconds exceeded", time_limit)

    env.close()

    if log_file is not None:
        with log_file.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trained AoE2DE agent.")
    parser.add_argument("--mission", required=True, help="Name of the campaign mission")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="Maximum runtime in seconds")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Optional JSON file to store state/action history")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_agent(args.mission, Path(args.weights), args.time_limit, args.log_file)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
