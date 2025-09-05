from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Entry point that dispatches to project scripts via subcommands."""
    parser = argparse.ArgumentParser(description="AoE2DE command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Treina o agente")
    subparsers.add_parser("run", help="Executa o agente em campanha")
    subparsers.add_parser("preprocess", help="Processa replays")

    args, remaining = parser.parse_known_args()

    if args.command == "train":
        from scripts import train_agent

        sys.argv = ["train_agent.py", *remaining]
        train_agent.main()
    elif args.command == "run":
        from scripts import run_agent

        sys.argv = ["run_agent.py", *remaining]
        run_agent.main()
    elif args.command == "preprocess":
        from scripts import preprocess_replays

        sys.argv = ["preprocess_replays.py", *remaining]
        preprocess_replays.main()


if __name__ == "__main__":
    main()
