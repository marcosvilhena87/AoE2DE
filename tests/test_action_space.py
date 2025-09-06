import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.action_space import (
    ACTION_ID_TO_COMPONENTS,
    COMPONENTS_TO_ACTION_ID,
    get_action_components,
    get_action_id,
)


def test_round_trip_action_mappings():
    """Each action ID should map to components and back again."""
    for action_id, components in ACTION_ID_TO_COMPONENTS.items():
        assert get_action_components(action_id) == components
        assert get_action_id(*components) == action_id

    # Also verify that the inverse mapping matches
    for components, action_id in COMPONENTS_TO_ACTION_ID.items():
        assert ACTION_ID_TO_COMPONENTS[action_id] == components
