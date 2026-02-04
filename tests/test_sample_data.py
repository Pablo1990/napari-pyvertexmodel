"""Tests for sample data module."""
import numpy as np

from napari_pyvertexmodel import make_sample_data


def test_make_sample_data_returns_list():
    """Test that make_sample_data returns a list."""
    result = make_sample_data()
    assert isinstance(result, list)
    assert len(result) > 0


def test_make_sample_data_format():
    """Test that make_sample_data returns correct format."""
    result = make_sample_data()
    
    # Should be a list of tuples
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 2
    
    data, kwargs = result[0]
    assert isinstance(data, np.ndarray)
    assert isinstance(kwargs, dict)


def test_make_sample_data_shape():
    """Test that make_sample_data returns correct shape."""
    result = make_sample_data()
    data, _ = result[0]
    
    # Should be 512x512
    assert data.shape == (512, 512)


def test_make_sample_data_values():
    """Test that make_sample_data returns valid values."""
    result = make_sample_data()
    data, _ = result[0]
    
    # Values should be in [0, 1] range for random data
    assert data.min() >= 0
    assert data.max() <= 1
