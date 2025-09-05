import pytest

@pytest.mark.skip(reason="utils module not implemented yet")
def test_add_example():
    """Example test for utils.add once implemented."""
    from src import utils
    assert utils.add(1, 2) == 3
