"""Tests for utils module."""
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from napari_pyvertexmodel.utils import (
    _add_surface_layer,
    _get_mesh,
    _create_surface_data,
)


@pytest.fixture
def mock_viewer():
    """Create a mock napari viewer."""
    viewer = Mock()
    viewer.dims = Mock()
    viewer.dims.set_current_step = Mock()
    viewer.layers = {}
    viewer.add_surface = Mock()
    return viewer


@pytest.fixture
def mock_cell():
    """Create a mock cell with VTK data."""
    cell = Mock()
    cell.ID = 1
    cell.AliveStatus = True
    
    # Mock VTK polydata
    vtk_poly = Mock()
    
    # Mock points
    points = Mock()
    points_data = Mock()
    points_data.return_value = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    points.GetData = points_data
    vtk_poly.GetPoints = Mock(return_value=points)
    
    # Mock faces (polys)
    polys = Mock()
    polys_data = Mock()
    # VTK format: [n_points, id1, id2, id3, ...]
    polys_data.return_value = np.array([3, 0, 1, 2], dtype=int)
    polys.GetData = polys_data
    vtk_poly.GetPolys = Mock(return_value=polys)
    
    # Mock scalars
    cell_data = Mock()
    scalars = Mock()
    scalars.return_value = np.array([0.001, 0.002, 0.003], dtype=float)
    cell_data.GetScalars = scalars
    vtk_poly.GetCellData = Mock(return_value=cell_data)
    
    cell.create_vtk = Mock(return_value=vtk_poly)
    return cell


@pytest.fixture
def mock_v_model(mock_cell):
    """Create a mock vertex model."""
    v_model = Mock()
    v_model.t = 0
    v_model.set = Mock()
    v_model.set.model_name = "test_model"
    v_model.geo = Mock()
    v_model.geo.Cells = [mock_cell]
    return v_model


def test_create_surface_data_basic(mock_cell, mock_v_model):
    """Test _create_surface_data with basic input."""
    with patch('napari_pyvertexmodel.utils.vtk_to_numpy') as mock_vtk_to_numpy:
        # Setup mock returns for vtk_to_numpy
        mock_vtk_to_numpy.side_effect = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),  # vertices
            np.array([3, 0, 1, 2], dtype=int),  # faces (raw VTK format)
            np.array([0.001, 0.002, 0.003], dtype=float),  # scalars
        ]
        
        layer_name, faces, scalars, vertices = _create_surface_data(
            mock_cell, mock_v_model, offset_indices=0
        )
        
        assert layer_name == "test_model_cell_1"
        assert vertices.shape == (3, 3)
        assert faces.shape == (1, 3)
        assert scalars.shape == (3,)
        assert np.array_equal(faces, np.array([[0, 1, 2]]))


def test_create_surface_data_with_offset(mock_cell, mock_v_model):
    """Test _create_surface_data with offset indices."""
    with patch('napari_pyvertexmodel.utils.vtk_to_numpy') as mock_vtk_to_numpy:
        mock_vtk_to_numpy.side_effect = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            np.array([3, 0, 1, 2], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
        ]
        
        offset = 10
        _, faces, _, _ = _create_surface_data(
            mock_cell, mock_v_model, offset_indices=offset
        )
        
        # Faces should be offset by 10
        assert np.array_equal(faces, np.array([[10, 11, 12]]))


def test_create_surface_data_scalar_padding(mock_cell, mock_v_model):
    """Test _create_surface_data when scalars need padding."""
    with patch('napari_pyvertexmodel.utils.vtk_to_numpy') as mock_vtk_to_numpy:
        mock_vtk_to_numpy.side_effect = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),  # 4 vertices
            np.array([3, 0, 1, 2], dtype=int),
            np.array([0.001, 0.002], dtype=float),  # Only 2 scalars
        ]
        
        _, _, scalars, vertices = _create_surface_data(
            mock_cell, mock_v_model, offset_indices=0
        )
        
        # Scalars should be padded to match vertices
        assert len(scalars) == vertices.shape[0]
        assert scalars[-2] == 0  # Padded values should be 0


def test_create_surface_data_scalar_truncation(mock_cell, mock_v_model):
    """Test _create_surface_data when scalars need truncation."""
    with patch('napari_pyvertexmodel.utils.vtk_to_numpy') as mock_vtk_to_numpy:
        mock_vtk_to_numpy.side_effect = [
            np.array([[0, 0, 0], [1, 0, 0]], dtype=float),  # 2 vertices
            np.array([3, 0, 1, 2], dtype=int),
            np.array([0.001, 0.002, 0.003, 0.004], dtype=float),  # 4 scalars
        ]
        
        _, _, scalars, vertices = _create_surface_data(
            mock_cell, mock_v_model, offset_indices=0
        )
        
        # Scalars should be truncated to match vertices
        assert len(scalars) == vertices.shape[0]


def test_get_mesh_single_cell(mock_v_model):
    """Test _get_mesh with a single cell."""
    with patch('napari_pyvertexmodel.utils._create_surface_data') as mock_create:
        mock_create.return_value = (
            "layer_name",
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        )
        
        faces, scalars, vertices = _get_mesh(mock_v_model)
        
        assert vertices.shape == (3, 3)
        assert faces.shape == (1, 3)
        assert scalars.shape == (3,)


def test_get_mesh_multiple_cells(mock_v_model):
    """Test _get_mesh with multiple cells."""
    # Create a second mock cell
    cell2 = Mock()
    cell2.ID = 2
    cell2.AliveStatus = True
    mock_v_model.geo.Cells = [mock_v_model.geo.Cells[0], cell2]
    
    with patch('napari_pyvertexmodel.utils._create_surface_data') as mock_create:
        # Return different data for each cell
        mock_create.side_effect = [
            (
                "layer1",
                np.array([[0, 1, 2]], dtype=int),
                np.array([0.001, 0.002, 0.003], dtype=float),
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            ),
            (
                "layer2",
                np.array([[3, 4, 5]], dtype=int),  # Offset by 3
                np.array([0.004, 0.005, 0.006], dtype=float),
                np.array([[1, 1, 0], [2, 1, 0], [1, 2, 0]], dtype=float),
            ),
        ]
        
        faces, scalars, vertices = _get_mesh(mock_v_model)
        
        # Should have 6 vertices, 2 faces, 6 scalars
        assert vertices.shape == (6, 3)
        assert faces.shape == (2, 3)
        assert scalars.shape == (6,)


def test_get_mesh_with_scaling(mock_v_model):
    """Test _get_mesh with input image dimensions for scaling."""
    with patch('napari_pyvertexmodel.utils._create_surface_data') as mock_create:
        mock_create.return_value = (
            "layer_name",
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float),
        )
        
        input_dims = [100, 100]
        faces, scalars, vertices = _get_mesh(mock_v_model, input_image_dims=input_dims)
        
        # Vertices should be scaled
        assert vertices.shape == (3, 3)
        # With input dims [100, 100] and bbox [0-10, 0-10], scale should be ~10
        assert vertices[1, 0] > 10  # X should be scaled up
        assert vertices[2, 1] > 10  # Y should be scaled up


def test_get_mesh_dead_cells_filtered(mock_v_model):
    """Test that dead cells are filtered out."""
    # Create a dead cell
    dead_cell = Mock()
    dead_cell.AliveStatus = None
    mock_v_model.geo.Cells = [mock_v_model.geo.Cells[0], dead_cell]
    
    with patch('napari_pyvertexmodel.utils._create_surface_data') as mock_create:
        mock_create.return_value = (
            "layer_name",
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        )
        
        faces, scalars, vertices = _get_mesh(mock_v_model)
        
        # Should only process the alive cell
        assert mock_create.call_count == 1


def test_add_surface_layer_new_layer(mock_viewer, mock_v_model):
    """Test adding a new surface layer."""
    with patch('napari_pyvertexmodel.utils._get_mesh') as mock_get_mesh:
        mock_get_mesh.return_value = (
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        )
        
        _add_surface_layer(mock_viewer, mock_v_model, add_time_point=False)
        
        # Should call add_surface
        mock_viewer.add_surface.assert_called_once()
        call_args = mock_viewer.add_surface.call_args
        
        # Check surface data
        surface_data = call_args[0][0]
        assert len(surface_data) == 3  # vertices, faces, scalars


def test_add_surface_layer_update_existing(mock_viewer, mock_v_model):
    """Test updating an existing surface layer."""
    # Add a mock layer to viewer
    layer_name = f"{mock_v_model.set.model_name}_all_cells"
    mock_layer = Mock()
    mock_viewer.layers = {layer_name: mock_layer}
    
    with patch('napari_pyvertexmodel.utils._get_mesh') as mock_get_mesh:
        mock_get_mesh.return_value = (
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        )
        
        _add_surface_layer(mock_viewer, mock_v_model, add_time_point=True)
        
        # Should update existing layer, not add new one
        mock_viewer.add_surface.assert_not_called()
        # Should update dims
        mock_viewer.dims.set_current_step.assert_called_once_with(0, 0)


def test_add_surface_layer_empty_vertices(mock_viewer, mock_v_model):
    """Test that no layer is added when there are no vertices."""
    with patch('napari_pyvertexmodel.utils._get_mesh') as mock_get_mesh:
        # Return empty arrays
        mock_get_mesh.return_value = (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([]).reshape(0, 3),
        )
        
        _add_surface_layer(mock_viewer, mock_v_model, add_time_point=False)
        
        # Should not add layer
        mock_viewer.add_surface.assert_not_called()


def test_add_surface_layer_with_image_dims(mock_viewer, mock_v_model):
    """Test adding surface layer with image dimensions."""
    with patch('napari_pyvertexmodel.utils._get_mesh') as mock_get_mesh:
        mock_get_mesh.return_value = (
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        )
        
        input_dims = [100, 100]
        _add_surface_layer(
            mock_viewer, mock_v_model, 
            input_image_dims=input_dims, 
            add_time_point=False
        )
        
        # Should pass image dims to _get_mesh
        mock_get_mesh.assert_called_once_with(mock_v_model, input_dims)


def test_get_mesh_zero_bbox_scaling(mock_v_model):
    """Test _get_mesh handles zero bounding box gracefully."""
    with patch('napari_pyvertexmodel.utils._create_surface_data') as mock_create:
        # All vertices have same X and Y coordinates (zero bbox)
        mock_create.return_value = (
            "layer_name",
            np.array([[0, 1, 2]], dtype=int),
            np.array([0.001, 0.002, 0.003], dtype=float),
            np.array([[5, 5, 0], [5, 5, 1], [5, 5, 2]], dtype=float),
        )
        
        input_dims = [100, 100]
        faces, scalars, vertices = _get_mesh(mock_v_model, input_image_dims=input_dims)
        
        # Should not crash or produce NaN values
        assert not np.any(np.isnan(vertices))
        assert vertices.shape == (3, 3)
