import os
import sys

import pytest

# Ensure src is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from macros.actions import MacroAction, available_actions
from macros.state import GameState


def test_no_actions_available():
    state = GameState(food=0, wood=0, age="Dark Age", buildings={"Town Center"})
    assert available_actions(state) == []


def test_train_villager_available():
    state = GameState(food=50, wood=0, age="Dark Age", buildings={"Town Center"})
    assert available_actions(state) == [MacroAction.TRAIN_VILLAGER]


def test_age_up_and_train_villager_available():
    state = GameState(food=500, wood=0, age="Dark Age", buildings={"Town Center"})
    actions = available_actions(state)
    assert set(actions) == {MacroAction.TRAIN_VILLAGER, MacroAction.AGE_UP_FEUDAL}


def test_build_archery_range_available():
    state = GameState(food=100, wood=200, age="Feudal Age", buildings={"Town Center"})
    actions = available_actions(state)
    assert set(actions) == {MacroAction.TRAIN_VILLAGER, MacroAction.BUILD_ARCHERY_RANGE}
