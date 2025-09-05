Command-line scripts for data processing and agent execution.

## Preprocess Replays

`preprocess_replays.py` converts raw `*.aoe2record` or `*.mgz` files into
JSON episodes. It relies on `ReplayParser` from `src/utils/replay_parser.py`
which reads a minimal binary format and extracts events with timestamps and
associated players. `.mgz` files are transparently decompressed using the
standard library's `gzip` module so no third-party dependencies are required.

### Usage

```bash
python scripts/preprocess_replays.py --input <replay_dir> --output <json_dir>
```

Optionally set `--workers` to control parallelism.
