"""Run a trained agent on a specific AoE2DE campaign mission."""

from __future__ import annotations

import argparse


def run_agent(mission_name: str) -> None:
    """Execute the trained agent for a given campaign mission.

    Parameters
    ----------
    mission_name:
        Human-readable name of the campaign mission to play.
    """
    # Placeholder implementation – this would launch the game and control the agent.
    print(f"Pretending to run agent on mission '{mission_name}'")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trained AoE2DE agent.")
    parser.add_argument("--mission", required=True, help="Name of the campaign mission")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_agent(args.mission)


if __name__ == "__main__":
    main()
