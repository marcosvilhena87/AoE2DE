import json
from pathlib import Path
from unittest import mock

import torch

import scripts.train_agent as ta


def test_train_agent_runs_training_loop(tmp_path, monkeypatch):
    # Prepare dummy config
    cfg = {
        "data_dir": "unused",
        "state_dim": 2,
        "num_actions": 3,
        "batch_size": 1,
        "tensorboard": False,
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg))
    output_dir = tmp_path / "out"

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            state = torch.zeros(1, cfg["state_dim"], dtype=torch.float32)
            action = torch.tensor(0, dtype=torch.long)
            return state, action

    monkeypatch.setattr(ta, "EpisodeDataset", lambda _p: DummyDataset())

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.lin = torch.nn.Linear(cfg["state_dim"], cfg["num_actions"])

        def forward(self, x):
            return self.lin(x[:, -1])

    monkeypatch.setattr(ta, "SimpleTransformer", DummyModel)

    opt_holder = {}

    class DummyOptim:
        def __init__(self, params, lr):
            opt_holder["instance"] = self
            self.step_count = 0

        def zero_grad(self):
            pass

        def step(self):
            self.step_count += 1

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

    monkeypatch.setattr(ta.optim, "Adam", DummyOptim)

    ta.train_agent(config_path, output_dir, epochs=2, use_gpu=False)

    opt = opt_holder["instance"]
    assert opt.step_count == 2

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert len(metrics) == 2
    assert metrics[0]["epoch"] == 1
