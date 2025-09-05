import sys
import types
import os

import numpy as np
import pytest


def test_load_transitions_mismatched_lengths(tmp_path, monkeypatch):
    # Create dummy imitation package so bc_train can be imported without the real dependency
    imitation = types.ModuleType("imitation")
    algorithms = types.ModuleType("imitation.algorithms")
    bc = types.ModuleType("imitation.algorithms.bc")
    algorithms.bc = bc
    data = types.ModuleType("imitation.data")
    types_mod = types.ModuleType("imitation.data.types")

    class Transitions:  # pragma: no cover - simple placeholder
        pass

    types_mod.Transitions = Transitions
    data.types = types_mod
    imitation.algorithms = algorithms
    imitation.data = data

    monkeypatch.setitem(sys.modules, "imitation", imitation)
    monkeypatch.setitem(sys.modules, "imitation.algorithms", algorithms)
    monkeypatch.setitem(sys.modules, "imitation.algorithms.bc", bc)
    monkeypatch.setitem(sys.modules, "imitation.data", data)
    monkeypatch.setitem(sys.modules, "imitation.data.types", types_mod)
    gymnasium = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")
    gymnasium.spaces = spaces
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces)

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from policies.bc_train import load_transitions

    # Create a trajectory file with mismatched obs and acts lengths
    path = tmp_path / "traj.npz"
    np.savez(path, obs=np.zeros((5, 3)), acts=np.zeros(4))

    with pytest.raises(ValueError, match="mismatched lengths"):
        load_transitions(tmp_path)
