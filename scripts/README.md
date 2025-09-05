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

## Train Agent

`train_agent.py` fits the imitation model using preprocessed episodes. It
expects a configuration file containing dataset paths and model
hyperparameters.

### Usage

```bash
python scripts/train_agent.py --config <cfg.json> --output-dir <out_dir> \
    --epochs 10 --seed 42
```

Key arguments:

- `--seed` – sets seeds for `random`, `numpy` and `torch` for reproducible
  training.
- `--use-gpu` – enable CUDA if available.

The configuration file must specify `data_dir`, `state_dim` and `num_actions`.
Optional fields configure batch size, model dimensions and early stopping
(`val_split` and `patience`). The best model is saved as `best_model.pt` in the
output directory.
