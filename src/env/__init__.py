"""Environment connector for interacting with Age of Empires II: DE.

This module provides a very small abstraction around grabbing game state
information and issuing hotkey commands.  The implementation is intentionally
minimal and relies on commonly available libraries such as :mod:`pyautogui`
for sending key presses and :mod:`pytesseract` for optical character
recognition.  The real game API is far more complex; this connector only
serves as a proof-of-concept for demonstrating how an agent could interface
with the game.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Mapping, Optional

# Mapping from macro actions (indices predicted by the model) to concrete
# sequences of hotkey commands to be sent to the game.  The mapping here is a
# tiny placeholder and should be extended to cover the full action space of the
# agent.
ACTION_MAP: Dict[int, Iterable[str]] = {
    0: ["space"],          # e.g. jump to idle villager
    1: ["a"],              # generic example action
    2: ["b", "v"],        # train a villager at the town centre
    3: ["b", "b"],        # build a house
    4: ["b", "a"],        # build a barracks
}


class AoE2DEEnvironment:
    """Light‑weight connector around the running AoE2DE game instance.

    The environment is responsible for reading the current game state via OCR
    and translating high‑level actions into actual hotkey presses.  It also
    exposes a small utility to check for completion of in‑game objectives, which
    may be implemented using custom logic.
    """

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    def read_state(self) -> List[float]:
        """Capture the current game state as a numeric vector.

        The default implementation grabs a screenshot of the active monitor and
        runs OCR over it.  The resulting text is then parsed for the four resource
        numbers (food, wood, gold, stone) and the current and maximum population.
        If parsing fails, missing fields default to ``0``.
        """

        try:
            import pyautogui  # type: ignore
            import pytesseract  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OCR dependencies are not available. Please install 'pyautogui' "
                "and 'pytesseract' to enable state capture") from exc

        screenshot = pyautogui.screenshot()
        text = pytesseract.image_to_string(screenshot)

        numbers = [int(n) for n in re.findall(r"\d+", text)]
        food = numbers[0] if len(numbers) > 0 else 0
        wood = numbers[1] if len(numbers) > 1 else 0
        gold = numbers[2] if len(numbers) > 2 else 0
        stone = numbers[3] if len(numbers) > 3 else 0

        pop_cur, pop_cap = 0, 0
        pop_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if pop_match:
            pop_cur, pop_cap = map(int, pop_match.groups())

        state = [
            float(food),
            float(wood),
            float(gold),
            float(stone),
            float(pop_cur),
            float(pop_cap),
        ]
        self.log.debug("Captured state %s from text %r", state, text)
        return state

    def send_action(self, action: int) -> None:
        """Translate an action index into concrete hotkeys and send them."""

        try:
            import pyautogui  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'pyautogui' package is required to send hotkeys") from exc

        keys = ACTION_MAP.get(action, [])
        for key in keys:
            pyautogui.press(key)
        self.log.debug("Sent action %s -> %s", action, list(keys))

    # The following two helpers are intentionally very small and serve mainly as
    # extension points for real‑world implementations.
    def objectives_completed(
        self,
        objectives: Optional[Mapping[str, float]] = None,
        state: Optional[List[float]] = None,
    ) -> bool:
        """Check whether the supplied mission ``objectives`` are satisfied.

        Parameters
        ----------
        objectives:
            Mapping of resource/population names to minimum required values.
        state:
            Optional pre‑computed state vector as returned by :meth:`read_state`.
            If not provided, :meth:`read_state` is invoked.
        """

        if objectives is None:
            objectives = {}
        if state is None:
            state = self.read_state()

        mapping = {
            "food": 0,
            "wood": 1,
            "gold": 2,
            "stone": 3,
            "population": 4,
            "pop_cap": 5,
        }

        for key, required in objectives.items():
            idx = mapping.get(key)
            if idx is None:
                continue
            if state[idx] < required:
                return False
        return True

    def close(self) -> None:  # pragma: no cover - simple resource release
        """Clean up any held resources (none for the placeholder)."""

        self.log.debug("Environment shut down")
