"""Utility to preprocess AoE2DE replays into training episodes."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from src.utils.replay_parser import ReplayParser, UnsupportedReplayFormat


logger = logging.getLogger(__name__)


def _iter_replay_files(directory: Path) -> Iterable[Path]:
    return (
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".aoe2record", ".mgz"}
    )


def _process_replay(replay_path: Path, output_dir: Path) -> None:
    parser = ReplayParser()
    try:
        metadata = parser.parse(replay_path)
        out_file = output_dir / f"{replay_path.stem}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Processed %s", replay_path.name)
    except (UnsupportedReplayFormat, ValueError) as exc:
        logger.warning("Skipping %s: %s", replay_path.name, exc)
    except Exception as exc:  # pragma: no cover - logging path
        logger.error("Failed to process %s: %s", replay_path, exc, exc_info=True)


def preprocess_replays(
    input_dir: str | Path, output_dir: str | Path, workers: int | None = None
) -> None:
    """Convert raw replay files into structured episodes."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    replay_files = list(_iter_replay_files(input_path))
    if not replay_files:
        logger.warning("No replay files found in %s", input_path)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_replay, replay, output_path): replay
            for replay in replay_files
        }
        for future in as_completed(futures):
            future.result()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess AoE2DE replay files.")
    parser.add_argument("--input", required=True, help="Directory with replay files")
    parser.add_argument(
        "--output", required=True, help="Directory to store processed episodes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers to use (default: number of CPUs)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()
    preprocess_replays(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()

