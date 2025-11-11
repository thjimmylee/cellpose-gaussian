"""Tests for core module."""
import pytest
from cellpose_gaussian.core import hello_world, add


def test_hello_world():
    """Test hello_world function."""
    result = hello_world()
    assert result == "Hello from cellpose-gaussian!"
    assert isinstance(result, str)


def test_add():
    """Test add function."""
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
    assert add(10, -5) == 5


def test_add_floats():
    """Test add function with floats."""
    assert add(2.5, 3.5) == 6.0
    assert add(1.1, 2.2) == pytest.approx(3.3)
