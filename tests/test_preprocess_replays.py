from __future__ import annotations

import logging
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.preprocess_replays import _process_replay
from src.utils.replay_parser import ReplayParser, UnsupportedReplayFormat


def test_process_replay_skips_unsupported(monkeypatch: pytest.MonkeyPatch, caplog, tmp_path: Path) -> None:
    replay = tmp_path / "bad.aoe2record"
    replay.write_bytes(b"junk")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _raise(_self, _path):
        raise UnsupportedReplayFormat("bad format")

    monkeypatch.setattr(ReplayParser, "parse", _raise)

    caplog.set_level(logging.WARNING)
    _process_replay(replay, out_dir)

    assert f"Skipping {replay.name}: bad format" in caplog.text


def test_process_replay_skips_value_error(monkeypatch: pytest.MonkeyPatch, caplog, tmp_path: Path) -> None:
    replay = tmp_path / "bad.aoe2record"
    replay.write_bytes(b"junk")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _raise(_self, _path):
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(ReplayParser, "parse", _raise)

    caplog.set_level(logging.WARNING)
    _process_replay(replay, out_dir)

    assert f"Skipping {replay.name}: checksum mismatch" in caplog.text

