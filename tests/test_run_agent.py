from pathlib import Path
from unittest import mock
import importlib
import sys

import torch

import scripts.train_agent as ta
sys.modules['train_agent'] = ta
ra = importlib.import_module('scripts.run_agent')


def test_run_agent_calls_environment(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __call__(self, x):
            return torch.tensor([[0.1, 0.9]])

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(ra, "load_model", lambda w, d: DummyModel())

    env = mock.MagicMock()
    env.read_state.return_value = [0.0]
    env.objectives_completed.return_value = True
    monkeypatch.setattr(ra, "AoE2DEEnvironment", lambda: env)
    monkeypatch.setattr(ra.time, "sleep", lambda _: None)

    ra.run_agent("m1", Path("weights.pt"), time_limit=1, log_file=None)

    env.read_state.assert_called_once()
    env.send_action.assert_called_once_with(1)
    env.close.assert_called_once()
