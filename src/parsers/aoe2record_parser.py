"""Utilities for parsing Age of Empires II: Definitive Edition replays.

This module uses the `mgz` library to decode replay files and export
training episodes in a JSON Lines format.  Each episode contains three
fields:

``state``
    Dictionary representing the decoded action payload.
``action_id``
    Numeric identifier of the action type.
``valid_action_mask``
    Placeholder list describing which actions are valid.  The mask is left
    empty as computing legal actions requires a full game simulator.

The exported episodes are saved under ``data/episodes`` with the same name
as the replay but a ``.jsonl`` extension.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List
import json

import mgz.fast as fast
from mgz.fast.enums import Operation


@dataclass
class Episode:
    """Representation of a single replay step."""

    state: Dict[str, Any]
    action_id: int
    valid_action_mask: List[int]


def parse_replay(replay_path: str) -> Iterator[Episode]:
    """Parse a replay file and yield :class:`Episode` objects.

    Parameters
    ----------
    replay_path:
        Path to a ``.mgz`` replay file.
    """

    path = Path(replay_path)
    with path.open("rb") as data:
        # The DE log format begins with a meta section followed by a start
        # block.  The ``fast`` helpers skip these initial bytes.
        fast.meta(data)
        fast.start(data)

        while True:
            op_type, payload = fast.operation(data)
            if op_type is Operation.ACTION:
                action_type, details = payload
                # Each action contains a ``sequence`` key which is not part of
                # the actionable state.  It is removed to keep the exported
                # state compact.
                state = dict(details)
                state.pop("sequence", None)
                yield Episode(state=state, action_id=action_type.value, valid_action_mask=[])
            elif op_type is Operation.POSTGAME:
                break


def export_episodes(replay_path: str, output_dir: str = "data/episodes") -> Path:
    """Export parsed episodes from ``replay_path`` to ``output_dir``.

    Returns the path of the generated ``.jsonl`` file.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(replay_path).stem + ".jsonl")

    with out_path.open("w", encoding="utf-8") as fh:
        for episode in parse_replay(replay_path):
            fh.write(json.dumps(asdict(episode)) + "\n")

    return out_path


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Convert an AoE2 replay into training episodes")
    parser.add_argument("replay", help="path to the .mgz replay file")
    parser.add_argument(
        "--output-dir",
        default="data/episodes",
        help="directory where the .jsonl file will be written",
    )
    args = parser.parse_args()
    output = export_episodes(args.replay, args.output_dir)
    print(f"wrote episodes to {output}")
