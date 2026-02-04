import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from napari_pyvertexmodel import napari_get_reader
from napari_pyvertexmodel._reader import (
    pkl_reader_function,
    npy_reader_function,
)


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    # we make the array an integer type to be compatible with the reader
    my_test_file = str(tmp_path / "myfile.npy")
    original_data = np.random.rand(20, 20).astype(np.int_)
    np.save(my_test_file, original_data)

    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_get_reader_pass(tmp_path):
    reader = napari_get_reader("fake.file")
    assert reader is None

    # the original_data is a float type, so the reader should return None
    my_test_file = str(tmp_path / "myfile.npy")
    original_data = np.random.rand(20, 20)
    np.save(my_test_file, original_data)

    reader = napari_get_reader(my_test_file)
    assert reader is None


def test_get_reader_with_list():
    """Test that reader handles list of paths."""
    paths = ["file1.npy", "file2.npy"]
    
    with patch("numpy.load") as mock_load:
        # Mock load to return integer array
        mock_load.return_value = np.array([[1, 2], [3, 4]], dtype=np.int_)
        reader = napari_get_reader(paths)
        
        # Should check only the first file
        assert callable(reader)


def test_get_reader_pkl_file(tmp_path):
    """Test that reader recognizes .pkl files."""
    pkl_file = tmp_path / "model.pkl"
    pkl_file.touch()  # Create empty file
    
    reader = napari_get_reader(str(pkl_file))
    assert callable(reader)
    assert reader == pkl_reader_function


def test_get_reader_npy_integer(tmp_path):
    """Test that reader recognizes integer .npy files."""
    npy_file = tmp_path / "data.npy"
    data = np.array([[1, 2], [3, 4]], dtype=np.int_)
    np.save(npy_file, data)
    
    reader = napari_get_reader(str(npy_file))
    assert callable(reader)
    assert reader == npy_reader_function


def test_get_reader_npy_float(tmp_path):
    """Test that reader rejects float .npy files."""
    npy_file = tmp_path / "data.npy"
    data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=float)
    np.save(npy_file, data)
    
    reader = napari_get_reader(str(npy_file))
    assert reader is None


def test_get_reader_invalid_npy(tmp_path):
    """Test that reader handles invalid .npy files."""
    npy_file = tmp_path / "invalid.npy"
    npy_file.write_text("not a numpy file")
    
    reader = napari_get_reader(str(npy_file))
    assert reader is None


def test_npy_reader_function_single_file(tmp_path):
    """Test npy_reader_function with a single file."""
    npy_file = tmp_path / "data.npy"
    original_data = np.array([[1, 2], [3, 4]], dtype=np.int_)
    np.save(npy_file, original_data)
    
    layer_data = npy_reader_function(str(npy_file))
    
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    data, kwargs, layer_type = layer_data[0]
    assert np.array_equal(data, original_data)
    assert isinstance(kwargs, dict)
    assert layer_type == "image"


def test_npy_reader_function_multiple_files(tmp_path):
    """Test npy_reader_function with multiple files."""
    files = []
    for i in range(3):
        npy_file = tmp_path / f"data{i}.npy"
        np.save(npy_file, np.array([[i, i+1], [i+2, i+3]], dtype=np.int_))
        files.append(str(npy_file))
    
    layer_data = npy_reader_function(files)
    
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    data, kwargs, layer_type = layer_data[0]
    assert data.shape == (3, 2, 2)
    assert layer_type == "image"


def test_pkl_reader_function_with_mock():
    """Test pkl_reader_function with mocked vertex model."""
    mock_path = "test.pkl"
    
    with patch('pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage.VertexModelVoronoiFromTimeImage') as MockVM, \
         patch('pyVertexModel.util.utils.load_state') as mock_load_state, \
         patch('napari_pyvertexmodel._reader._get_mesh') as mock_get_mesh:
        
        # Setup mock vertex model
        mock_v_model = Mock()
        mock_v_model.set.model_name = "test_model"
        MockVM.return_value = mock_v_model
        
        # Setup mock mesh data
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        scalars = np.array([0.001, 0.002, 0.003], dtype=float)
        mock_get_mesh.return_value = (faces, scalars, vertices)
        
        layer_data = pkl_reader_function(mock_path)
        
        # Verify the function was called correctly
        MockVM.assert_called_once()
        mock_load_state.assert_called_once_with(mock_v_model, mock_path)
        
        # Check returned data
        assert isinstance(layer_data, list)
        assert len(layer_data) == 1
        surface_data, kwargs, layer_type = layer_data[0]
        assert layer_type == "surface"
        assert "name" in kwargs
        assert "colormap" in kwargs


def test_pkl_reader_function_multiple_files():
    """Test pkl_reader_function with multiple files."""
    mock_paths = ["test1.pkl", "test2.pkl"]
    
    with patch('pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage.VertexModelVoronoiFromTimeImage') as MockVM, \
         patch('pyVertexModel.util.utils.load_state') as mock_load_state, \
         patch('napari_pyvertexmodel._reader._get_mesh') as mock_get_mesh:
        
        mock_v_model = Mock()
        mock_v_model.set.model_name = "test_model"
        MockVM.return_value = mock_v_model
        
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        scalars = np.array([0.001, 0.002, 0.003], dtype=float)
        mock_get_mesh.return_value = (faces, scalars, vertices)
        
        layer_data = pkl_reader_function(mock_paths)
        
        # Should have two layers
        assert len(layer_data) == 2
        assert mock_load_state.call_count == 2


def test_pkl_reader_function_error_handling():
    """Test pkl_reader_function handles errors gracefully."""
    mock_path = "bad.pkl"
    
    with patch('pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage.VertexModelVoronoiFromTimeImage') as MockVM, \
         patch('pyVertexModel.util.utils.load_state') as mock_load_state:
        
        # Make load_state raise an exception
        mock_load_state.side_effect = Exception("Load failed")
        mock_v_model = Mock()
        MockVM.return_value = mock_v_model
        
        layer_data = pkl_reader_function(mock_path)
        
        # Should return empty list on error
        assert layer_data == []


def test_pkl_reader_function_empty_vertices():
    """Test pkl_reader_function with empty vertex data."""
    mock_path = "empty.pkl"
    
    with patch('pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage.VertexModelVoronoiFromTimeImage') as MockVM, \
         patch('pyVertexModel.util.utils.load_state') as mock_load_state, \
         patch('napari_pyvertexmodel._reader._get_mesh') as mock_get_mesh:
        
        mock_v_model = Mock()
        mock_v_model.set.model_name = "test_model"
        MockVM.return_value = mock_v_model
        
        # Return empty mesh
        mock_get_mesh.return_value = (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([]).reshape(0, 3),
        )
        
        layer_data = pkl_reader_function(mock_path)
        
        # Should return empty list when no vertices
        assert layer_data == []