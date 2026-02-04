"""Tests for widget module."""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

# Skip widget tests that require Qt
pytestmark = pytest.mark.skip(reason="Widget tests require Qt GUI which crashes in headless mode")


def test_widget_module_imports():
    """Test that widget module can be imported."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    assert Run3dVertexModel is not None
    
    # Test that important constants exist
    from napari_pyvertexmodel._widget import DEFAULT_VERTEX_MODEL_OPTION
    assert DEFAULT_VERTEX_MODEL_OPTION == 'wing_disc_equilibrium'


