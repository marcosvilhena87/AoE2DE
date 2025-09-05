"""Simple utilities for parsing macro action strings.

This module currently provides a minimal parser that converts human readable
strings into the :class:`~macros.actions.MacroAction` enum used throughout the
project.  The goal is to keep the mapping logic in one place so both the
environment and tests rely on the same behaviour.
"""

from __future__ import annotations

from macros.actions import MacroAction


def parse_macro_action(text: str) -> MacroAction:
    """Parse ``text`` into a :class:`~macros.actions.MacroAction`.

    Parameters
    ----------
    text:
        Human readable name of the action, e.g. ``"train villager"``.

    Returns
    -------
    macros.actions.MacroAction
        The corresponding enum value.

    Raises
    ------
    ValueError
        If ``text`` does not correspond to a known action.
    """

    normalized = text.strip().lower()
    mapping = {
        "train villager": MacroAction.TRAIN_VILLAGER,
        "age up": MacroAction.AGE_UP_FEUDAL,
        "build archery range": MacroAction.BUILD_ARCHERY_RANGE,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:  # pragma: no cover - error path
        raise ValueError(f"Unknown action: {text}") from exc


__all__ = ["parse_macro_action"]

