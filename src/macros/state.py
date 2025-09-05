from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class GameState:
    """Simplified representation of the game state."""

    food: int = 0
    wood: int = 0
    age: str = ""
    buildings: Set[str] = field(default_factory=set)
