"""Tests for widget module."""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


def test_widget_module_imports():
    """Test that widget module can be imported."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    assert Run3dVertexModel is not None
    
    # Test that important constants exist
    from napari_pyvertexmodel._widget import DEFAULT_VERTEX_MODEL_OPTION, PROJECT_DIRECTORY
    assert DEFAULT_VERTEX_MODEL_OPTION == 'wing_disc_equilibrium'
    assert PROJECT_DIRECTORY is not None


@pytest.fixture
def make_napari_viewer(qapp):
    """Fixture that creates a napari viewer for testing."""
    pytest.importorskip("napari")
    from napari import Viewer
    
    viewer = Viewer(show=False)
    yield viewer
    viewer.close()


def test_widget_initialization(make_napari_viewer, qtbot):
    """Test that widget initializes correctly."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    assert widget._viewer == viewer
    assert viewer.dims.ndisplay == 3
    assert widget.v_model is None


def test_widget_has_required_widgets(make_napari_viewer, qtbot):
    """Test that widget has all required sub-widgets."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Check that key widgets exist
    assert widget._image_layer_combo is not None
    assert widget._tissue_number_of_cells_slider is not None
    assert widget._tissue_height_slider is not None
    assert widget._lambda_volume_slider is not None
    assert widget._run_button is not None
    assert widget._load_simulation_button is not None


def test_widget_slider_defaults(make_napari_viewer, qtbot):
    """Test that sliders have correct default values."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    assert widget._tissue_number_of_cells_slider.value == 20
    assert widget._tissue_number_of_cells_slider.min == 10
    assert widget._tissue_number_of_cells_slider.max == 200
    
    assert widget._tissue_height_slider.value == 50
    assert widget._tissue_height_slider.min == 0.01
    assert widget._tissue_height_slider.max == 100.0
    
    assert widget._lambda_volume_slider.value == 1.0
    assert widget._volume_reference_slider.value == 1.0


def test_run_model_without_model(make_napari_viewer, qtbot, capsys):
    """Test running model without loading one first."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    widget._run_model()
    
    captured = capsys.readouterr()
    assert "Error: No model loaded" in captured.out


def test_update_sliders_from_model(make_napari_viewer, qtbot):
    """Test updating slider values from model."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Create a mock model
    mock_model = Mock()
    mock_model.set.TotalCells = 50
    mock_model.set.CellHeight = 75.0
    mock_model.set.lambdaV = 2.0
    mock_model.set.lambdaS1 = 0.7
    mock_model.set.lambdaS3 = 0.8
    mock_model.set.lambdaS2 = 0.3
    mock_model.set.ref_V0 = 1.5
    mock_model.set.ref_A0 = 0.95
    mock_model.set.kSubstrate = 0.2
    mock_model.set.tend = 2.0
    
    widget.v_model = mock_model
    widget._update_sliders_from_model()
    
    assert widget._tissue_number_of_cells_slider.value == 50
    assert widget._tissue_height_slider.value == 75.0
    assert widget._lambda_volume_slider.value == 2.0
    assert widget._lambda_surface_top_slider.value == 0.7


def test_update_model_from_sliders(make_napari_viewer, qtbot):
    """Test updating model from slider values."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Create a mock model
    mock_model = Mock()
    mock_model.set = Mock()
    widget.v_model = mock_model
    
    # Set slider values
    widget._lambda_volume_slider.value = 3.0
    widget._lambda_surface_top_slider.value = 0.6
    widget._volume_reference_slider.value = 2.0
    
    widget._update_model_from_sliders()
    
    assert mock_model.set.lambdaV == 3.0
    assert mock_model.set.lambdaS1 == 0.6
    assert mock_model.set.ref_V0 == 2.0


def test_display_advanced_params_show(make_napari_viewer, qtbot):
    """Test showing advanced parameters."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Initially advanced params should not be in widget
    assert widget._lambda_r_slider not in widget
    
    # Show advanced params
    widget._show_advanced_params_checkbox.value = True
    widget._display_advanced_params()
    
    # Advanced params should now be in widget
    assert widget._lambda_r_slider in widget
    assert widget._viscosity_slider in widget
    assert widget._remodelling_checkbox in widget


def test_display_advanced_params_hide(make_napari_viewer, qtbot):
    """Test hiding advanced parameters."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # First show them
    widget._show_advanced_params_checkbox.value = True
    widget._display_advanced_params()
    
    # Then hide them
    widget._show_advanced_params_checkbox.value = False
    widget._display_advanced_params()
    
    # Advanced params should be removed
    assert widget._lambda_r_slider not in widget
    assert widget._viscosity_slider not in widget


def test_load_simulation_no_file(make_napari_viewer, qtbot):
    """Test loading simulation with no file selected - file value should be empty string."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Set empty string (default value for FileEdit)
    widget._load_simulation_input.value = ""
    widget._load_simulation()
    
    # Should return early without error
    assert widget.v_model is None


def test_load_simulation_invalid_extension(make_napari_viewer, qtbot, capsys):
    """Test loading simulation with invalid file extension."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    widget._load_simulation_input.value = "/tmp/test.txt"
    widget._load_simulation()
    
    captured = capsys.readouterr()
    assert "valid .pkl file" in captured.out


def test_image_layer_load_with_layer(make_napari_viewer, qtbot):
    """Test loading image layer with a layer added."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    
    # Add a labels layer to the viewer first
    labels_data = np.random.randint(0, 10, (50, 50))
    viewer.add_labels(labels_data, name="test_labels")
    
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Now the combo box should have choices
    # We can test that initialization worked
    assert widget._image_layer_combo is not None


def test_create_temp_folder(make_napari_viewer, qtbot):
    """Test creating temporary folder."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Create mock model
    mock_model = Mock()
    mock_model.create_temporary_folder = Mock()
    widget.v_model = mock_model
    
    widget._create_temp_folder()
    
    mock_model.create_temporary_folder.assert_called_once()


def test_widget_cleanup(make_napari_viewer, qtbot):
    """Test widget cleanup on deletion."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Create a mock temp directory
    mock_temp_dir = Mock()
    mock_temp_dir.cleanup = Mock()
    widget._temp_dir = mock_temp_dir
    
    # Call destructor
    widget.__del__()
    
    # Cleanup should be called
    mock_temp_dir.cleanup.assert_called_once()


def test_widget_cleanup_no_temp_dir(make_napari_viewer, qtbot):
    """Test widget cleanup when no temp dir exists."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Should not raise error
    widget.__del__()


def test_widget_cleanup_with_error(make_napari_viewer, qtbot):
    """Test widget cleanup handles errors gracefully."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    mock_temp_dir = Mock()
    mock_temp_dir.cleanup.side_effect = Exception("Cleanup failed")
    widget._temp_dir = mock_temp_dir
    
    # Should not raise error
    widget.__del__()


def test_advanced_slider_defaults(make_napari_viewer, qtbot):
    """Test advanced slider default values."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    assert widget._lambda_r_slider.value == 8e-7
    assert widget._viscosity_slider.value == 0.07
    assert widget._remodelling_checkbox.value == False
    assert widget._ablation_checkbox.value == False
    assert widget._cells_to_ablate_slider.value == 0


def test_update_model_with_advanced_params(make_napari_viewer, qtbot):
    """Test updating model with advanced parameters enabled."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    mock_model = Mock()
    mock_model.set = Mock()
    widget.v_model = mock_model
    
    # Enable advanced params
    widget._show_advanced_params_checkbox.value = True
    widget._lambda_r_slider.value = 1e-6
    widget._viscosity_slider.value = 0.08
    widget._remodelling_checkbox.value = True
    widget._ablation_checkbox.value = True
    widget._cells_to_ablate_slider.value = 3
    
    widget._update_model_from_sliders()
    
    assert mock_model.set.lambdaR == 1e-6
    assert mock_model.set.nu == 0.08
    assert mock_model.set.RemodelCells == True
    assert mock_model.set.AblateCells == True
    assert np.array_equal(mock_model.set.CellsToAblate, np.arange(3))


def test_update_sliders_from_model_with_advanced_params(make_napari_viewer, qtbot):
    """Test updating sliders with advanced parameters enabled."""
    from napari_pyvertexmodel._widget import Run3dVertexModel
    
    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    
    # Enable advanced params
    widget._show_advanced_params_checkbox.value = True
    widget._display_advanced_params()
    
    # Create a mock model with advanced params
    mock_model = Mock()
    mock_model.set.TotalCells = 30
    mock_model.set.CellHeight = 60.0
    mock_model.set.lambdaV = 1.5
    mock_model.set.lambdaS1 = 0.6
    mock_model.set.lambdaS3 = 0.7
    mock_model.set.lambdaS2 = 0.2
    mock_model.set.ref_V0 = 1.2
    mock_model.set.ref_A0 = 0.9
    mock_model.set.kSubstrate = 0.15
    mock_model.set.tend = 1.5
    mock_model.set.lambdaR = 1e-6
    mock_model.set.nu = 0.08
    mock_model.set.RemodelCells = True
    mock_model.set.AblateCells = True
    mock_model.geo.cells_to_ablate = [1, 2, 3]
    
    widget.v_model = mock_model
    widget._update_sliders_from_model()
    
    assert widget._lambda_r_slider.value == 1e-6
    assert widget._viscosity_slider.value == 0.08
    assert widget._remodelling_checkbox.value == True
    assert widget._ablation_checkbox.value == True
    assert widget._cells_to_ablate_slider.value == 3


