"""Action space definitions for mapping action IDs to (verb, argument).

This module defines a small, illustrative mapping between integer action
identifiers and the corresponding verb/argument pair that describes the action.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Mapping from action ID to (verb, argument)
ACTION_ID_TO_COMPONENTS: Dict[int, Tuple[str, str]] = {
    0: ("move", "north"),
    1: ("move", "south"),
    2: ("attack", "melee"),
    3: ("build", "house"),
}

# Automatically create the inverse mapping
COMPONENTS_TO_ACTION_ID: Dict[Tuple[str, str], int] = {
    components: action_id for action_id, components in ACTION_ID_TO_COMPONENTS.items()
}

def get_action_components(action_id: int) -> Tuple[str, str]:
    """Return the (verb, argument) pair for a given action ID."""
    return ACTION_ID_TO_COMPONENTS[action_id]

def get_action_id(verb: str, argument: str) -> int:
    """Return the action ID for a given verb and argument."""
    return COMPONENTS_TO_ACTION_ID[(verb, argument)]
