from __future__ import annotations

import gzip
import struct
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils.replay_parser import ReplayParser


def _write_mgz(file: Path, events: list[tuple[int, str, str]]) -> None:
    with gzip.open(file, "wb") as f:
        f.write(struct.pack("<I", len(events)))
        for ts, player, event in events:
            player_bytes = player.encode("utf-8")
            event_bytes = event.encode("utf-8")
            f.write(struct.pack("<I", ts))
            f.write(bytes([len(player_bytes)]))
            f.write(player_bytes)
            f.write(bytes([len(event_bytes)]))
            f.write(event_bytes)


def test_parse_valid_mgz(tmp_path: Path) -> None:
    events = [(1000, "Alice", "move"), (2000, "Bob", "attack")]
    path = tmp_path / "game.mgz"
    _write_mgz(path, events)

    parser = ReplayParser()
    result = parser.parse(path)
    assert result == [
        {"timestamp": 1000, "player": "Alice", "event": "move"},
        {"timestamp": 2000, "player": "Bob", "event": "attack"},
    ]


def test_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.aoe2record"
    # Write event count and a partial event (timestamp only)
    with path.open("wb") as f:
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 1234))

    parser = ReplayParser()
    with pytest.raises(ValueError):
        parser.parse(path)
