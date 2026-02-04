"""Tests for writer module."""
import numpy as np
import pytest

from napari_pyvertexmodel import write_single_image, write_multiple


def test_write_single_image_returns_path():
    """Test that write_single_image returns a list with the path."""
    path = "/tmp/test.png"
    data = np.random.rand(10, 10)
    meta = {}
    
    result = write_single_image(path, data, meta)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == path


def test_write_single_image_with_metadata():
    """Test write_single_image with metadata."""
    path = "/tmp/test.png"
    data = np.random.rand(10, 10)
    meta = {"name": "test_layer", "opacity": 0.5}
    
    result = write_single_image(path, data, meta)
    
    assert result == [path]


def test_write_single_image_with_different_data_types():
    """Test write_single_image with different data types."""
    path = "/tmp/test.png"
    
    # Test with integer data
    data_int = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    result = write_single_image(path, data_int, {})
    assert result == [path]
    
    # Test with float data
    data_float = np.random.rand(10, 10)
    result = write_single_image(path, data_float, {})
    assert result == [path]


def test_write_multiple_returns_path():
    """Test that write_multiple returns a list with the path."""
    path = "/tmp/test_multi.zarr"
    data = [
        (np.random.rand(10, 10), {}, "image"),
        (np.random.rand(10, 10), {}, "labels"),
    ]
    
    result = write_multiple(path, data)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == path


def test_write_multiple_single_layer():
    """Test write_multiple with a single layer."""
    path = "/tmp/test.zarr"
    data = [(np.random.rand(5, 5), {"name": "layer1"}, "image")]
    
    result = write_multiple(path, data)
    
    assert result == [path]


def test_write_multiple_empty_list():
    """Test write_multiple with empty list."""
    path = "/tmp/test.zarr"
    data = []
    
    result = write_multiple(path, data)
    
    assert result == [path]


def test_write_multiple_different_layer_types():
    """Test write_multiple with different layer types."""
    path = "/tmp/test.zarr"
    data = [
        (np.random.rand(10, 10), {}, "image"),
        (np.random.randint(0, 10, (10, 10)), {}, "labels"),
        (np.random.rand(10, 10) > 0.5, {}, "shapes"),
    ]
    
    result = write_multiple(path, data)
    
    assert result == [path]
