from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from train_agent import TrainConfig, load_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(data))
    return cfg_path


def test_load_config_success(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = {"data_dir": str(data_dir), "state_dim": 4, "num_actions": 3}
    cfg_path = _write_config(tmp_path, cfg)
    cfg_obj = load_config(cfg_path)
    assert cfg_obj.state_dim == 4
    assert cfg_obj.batch_size == 32


def test_load_config_missing_field(tmp_path: Path) -> None:
    cfg = {"state_dim": 4, "num_actions": 3}
    cfg_path = _write_config(tmp_path, cfg)
    with pytest.raises(TypeError):
        load_config(cfg_path)


def test_load_config_invalid_batch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = {
        "data_dir": str(data_dir),
        "state_dim": 4,
        "num_actions": 3,
        "batch_size": 0,
    }
    cfg_path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_load_config_bad_path(tmp_path: Path) -> None:
    cfg = {"data_dir": str(tmp_path / "missing"), "state_dim": 4, "num_actions": 3}
    cfg_path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError):
        load_config(cfg_path)
