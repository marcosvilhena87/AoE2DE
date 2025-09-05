from __future__ import annotations

from pathlib import Path
import sys
import zlib

import mgz
from construct import ConstructError
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.replay_parser import ReplayParser, UnsupportedReplayFormat


def _make_header(valid_checksum: bool = True) -> dict:
    data = b"sample"
    computed = zlib.crc32(data) & 0xFFFFFFFF
    crc = computed if valid_checksum else computed ^ 0xFFFFFFFF
    return {
        "version": "DE",
        "map_info": {"size_x": 120, "size_y": 100},
        "de": {
            "players": [
                {"name": {"value": b"Alice"}, "civ_id": 1, "color_id": 1},
                {"name": {"value": b"Bob"}, "civ_id": 2, "color_id": 2},
            ],
            "rms_strings": {"strings": [{"crc": crc, "string": {"value": data}}]},
            "other_strings": [],
        },
    }


def test_parse_metadata(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "game.aoe2record"
    path.write_bytes(b"fake")
    header = _make_header()
    monkeypatch.setattr(type(mgz.header), "parse_stream", lambda self, fh: header)

    parser = ReplayParser()
    meta = parser.parse(path)

    assert meta["game_version"] == "DE"
    assert meta["map"] == {"size_x": 120, "size_y": 100}
    assert [p["name"] for p in meta["players"]] == ["Alice", "Bob"]


def test_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.aoe2record"
    path.write_bytes(b"fake")
    header = _make_header(valid_checksum=False)
    monkeypatch.setattr(type(mgz.header), "parse_stream", lambda self, fh: header)

    parser = ReplayParser()
    with pytest.raises(ValueError):
        parser.parse(path)


def test_unsupported_replay(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.aoe2record"
    path.write_bytes(b"fake")

    def _raise(_fh):
        raise ConstructError("bad")

    monkeypatch.setattr(type(mgz.header), "parse_stream", lambda self, fh: _raise(fh))

    parser = ReplayParser()
    with pytest.raises(UnsupportedReplayFormat):
        parser.parse(path)

