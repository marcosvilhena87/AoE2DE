from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from src.env import ACTION_MAP, AoE2DEEnvironment


def test_send_action_triggers_hotkeys(monkeypatch: pytest.MonkeyPatch) -> None:
    pressed: list[str] = []

    def fake_press(key: str) -> None:
        pressed.append(key)

    fake_pyautogui = SimpleNamespace(press=fake_press)
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)

    env = AoE2DEEnvironment()
    env.send_action(2)
    assert pressed == list(ACTION_MAP[2])


def test_read_state_parses_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyautogui = SimpleNamespace(screenshot=lambda: object())
    fake_text = "100 200 300 400\n10/50"
    fake_pytesseract = SimpleNamespace(image_to_string=lambda img: fake_text)

    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    env = AoE2DEEnvironment()
    state = env.read_state()
    assert state == [100.0, 200.0, 300.0, 400.0, 10.0, 50.0]


def test_objectives_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    env = AoE2DEEnvironment()
    state = [100.0, 50.0, 25.0, 10.0, 30.0, 40.0]
    assert env.objectives_completed({"food": 100, "population": 30}, state)
    assert not env.objectives_completed({"wood": 100}, state)

    monkeypatch.setattr(env, "read_state", lambda: state)
    assert env.objectives_completed({"food": 100, "population": 30})
