import numpy as np


def test_numpy_import():
    """Test that NumPy is installed and can be imported."""
    assert np is not None
    print(f"NumPy version: {np.__version__}")


def test_numpy_basic_functionality():
    """Test basic NumPy functionality."""
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    assert arr.mean() == 2.0
