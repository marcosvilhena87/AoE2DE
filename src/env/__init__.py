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
from typing import Dict, Iterable, List

# Mapping from macro actions (indices predicted by the model) to concrete
# sequences of hotkey commands to be sent to the game.  The mapping here is a
# tiny placeholder and should be extended to cover the full action space of the
# agent.
ACTION_MAP: Dict[int, Iterable[str]] = {
    0: ["space"],  # e.g. jump to idle villager
    1: ["a"],      # generic example action
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
        runs OCR over it.  The resulting text is converted into a very small set
        of numerical features.  Real agents would parse individual numbers from
        the HUD instead of using this crude representation.
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

        # For demonstration purposes the state is simply the length of the OCR'd
        # text.  A proper implementation would parse resources, unit counts, etc.
        state = [float(len(text))]
        self.log.debug("Captured state %s", state)
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
    def objectives_completed(self) -> bool:
        """Return ``True`` when mission objectives are satisfied."""

        return False

    def close(self) -> None:  # pragma: no cover - simple resource release
        """Clean up any held resources (none for the placeholder)."""

        self.log.debug("Environment shut down")
