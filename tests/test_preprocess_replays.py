import json
from pathlib import Path

from scripts.preprocess_replays import preprocess_replays


def test_preprocess_replays_creates_episode(tmp_path):
    input_dir = tmp_path / "replays"
    input_dir.mkdir()
    replay_file = input_dir / "game1.aoe2record"
    replay_file.write_bytes(b"1234")

    output_dir = tmp_path / "episodes"

    preprocess_replays(input_dir, output_dir, workers=1)

    episode_file = output_dir / "game1.json"
    assert episode_file.exists()

    data = json.loads(episode_file.read_text())
    assert data["events"][0]["bytes"] == 4
