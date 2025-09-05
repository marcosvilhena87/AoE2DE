from __future__ import annotations

"""Macro-level actions and utilities.

This module defines high-level macro actions available in the game and
provides a utility to determine which of those actions are currently
available given a simplified game state.
"""

from enum import Enum, auto
from typing import List, Set

from .state import GameState


class MacroAction(Enum):
    """Enumeration of simplified macro actions."""

    TRAIN_VILLAGER = auto()
    AGE_UP_FEUDAL = auto()
    BUILD_ARCHERY_RANGE = auto()


def available_actions(state: GameState) -> List[MacroAction]:
    """Return a list of available macro actions for a given state."""

    food: int = state.food
    wood: int = state.wood
    age: str = state.age
    buildings: Set[str] = set(state.buildings)

    actions: List[MacroAction] = []

    if food >= 50 and "Town Center" in buildings:
        actions.append(MacroAction.TRAIN_VILLAGER)

    if age == "Dark Age" and food >= 500:
        actions.append(MacroAction.AGE_UP_FEUDAL)

    if age == "Feudal Age" and wood >= 175:
        actions.append(MacroAction.BUILD_ARCHERY_RANGE)

    return actions
