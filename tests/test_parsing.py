import os
import sys

import pytest

# Ensure src is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from parsing import parse_macro_action
from macros.actions import MacroAction


def test_parse_valid_actions():
    assert parse_macro_action("train villager") == MacroAction.TRAIN_VILLAGER
    assert parse_macro_action("age up") == MacroAction.AGE_UP_FEUDAL
    assert parse_macro_action("build archery range") == MacroAction.BUILD_ARCHERY_RANGE


def test_parse_invalid_action():
    with pytest.raises(ValueError):
        parse_macro_action("fly to the moon")
