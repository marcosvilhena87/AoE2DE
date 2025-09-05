"""Utility to preprocess AoE2DE replays into training episodes."""

from __future__ import annotations

import argparse
from pathlib import Path


def preprocess_replays(input_dir: str | Path, output_dir: str | Path) -> None:
    """Convert raw replay files into structured episodes.

    Parameters
    ----------
    input_dir:
        Directory containing raw replay files.
    output_dir:
        Directory where processed episodes will be written.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    # Placeholder implementation – real preprocessing would parse replay files.
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Pretending to preprocess replays from {input_path} to {output_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess AoE2DE replay files.")
    parser.add_argument("--input", required=True, help="Directory with replay files")
    parser.add_argument(
        "--output", required=True, help="Directory to store processed episodes"
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    preprocess_replays(args.input, args.output)


if __name__ == "__main__":
    main()
