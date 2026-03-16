"""Tests for widget module."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


def test_widget_module_imports():
    """Test that widget module can be imported."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    assert Run3dVertexModel is not None

    # Test that important constants exist
    from napari_pyvertexmodel._widget import (
        DEFAULT_VERTEX_MODEL_OPTION,
        PROJECT_DIRECTORY,
    )

    assert DEFAULT_VERTEX_MODEL_OPTION == "wing_disc_equilibrium"
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
    assert widget._cancel_button is not None


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


def test_combo_accepts_image_layer(make_napari_viewer, qtbot):
    """Test that the combo box accepts Image layers (Image is subclass of Layer)."""
    import napari.layers

    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    image_data = np.random.rand(50, 50).astype(np.float32)
    viewer.add_image(image_data, name="test_image")

    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    assert widget._image_layer_combo is not None
    # The annotation should be Layer, which accepts Image as a subclass
    annotation = widget._image_layer_combo.annotation
    assert issubclass(napari.layers.Image, annotation)


def test_combo_accepts_labels_layer(make_napari_viewer, qtbot):
    """Test that the combo box accepts Labels layers (Labels is subclass of Layer)."""
    import napari.layers

    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    labels_data = np.random.randint(0, 10, (50, 50))
    viewer.add_labels(labels_data, name="test_labels")

    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    assert widget._image_layer_combo is not None
    # The annotation should be Layer, which accepts Labels as a subclass
    annotation = widget._image_layer_combo.annotation
    assert issubclass(napari.layers.Labels, annotation)


def test_image_layer_load_labels_converts_to_binary(make_napari_viewer, qtbot):
    """Test that loading a Labels layer converts it to a binary image."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer

    # Create labels with multiple label values
    labels_data = np.array([[0, 1, 2], [3, 0, 4], [0, 5, 0]], dtype=np.int_)
    labels_layer = viewer.add_labels(labels_data, name="test_labels")

    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    # Point the combo to the labels layer
    widget._image_layer_combo.value = labels_layer

    captured_label_data = {}

    mock_v_model = MagicMock()
    mock_v_model.set.model_name = "test_labels"
    mock_v_model.create_temporary_folder.return_value = "/tmp/mock"

    def capture_initialize(data):
        captured_label_data["data"] = data

    mock_v_model.initialize.side_effect = capture_initialize

    mock_worker = MagicMock()

    def synchronous_thread_worker(func):
        def creator(*args, **kwargs):
            result = func()
            # If _run_load is a generator, exhaust it so its body runs
            if hasattr(result, "__next__"):
                try:
                    while True:
                        next(result)
                except StopIteration:
                    pass
            return mock_worker

        return creator

    with (
        patch(
            "napari_pyvertexmodel._widget.thread_worker",
            synchronous_thread_worker,
        ),
        patch(
            "napari_pyvertexmodel._widget.VertexModelVoronoiFromTimeImage",
            return_value=mock_v_model,
        ),
        patch("napari_pyvertexmodel._widget._add_surface_layer"),
        patch("napari_pyvertexmodel._widget.progress"),
    ):
        widget._image_layer_load()

    # initialize should have been called with a binary image
    assert "data" in captured_label_data
    result = captured_label_data["data"]
    # The code passes `image_layer.data == 0`, so True where background (label=0)
    expected = labels_data == 0
    np.testing.assert_array_equal(result, expected)
    # Binary: only False/True (0/1) values
    assert set(np.unique(result)).issubset({False, True})


def test_image_layer_load_image_uses_data_directly(make_napari_viewer, qtbot):
    """Test that loading an Image layer passes data through unchanged."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer

    image_data = np.random.rand(50, 50).astype(np.float32)
    image_layer = viewer.add_image(image_data, name="test_image")

    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._image_layer_combo.value = image_layer

    captured_label_data = {}

    mock_v_model = MagicMock()
    mock_v_model.set.model_name = "test_image"
    mock_v_model.create_temporary_folder.return_value = "/tmp/mock"

    def capture_initialize(data):
        captured_label_data["data"] = data

    mock_v_model.initialize.side_effect = capture_initialize

    mock_worker = MagicMock()

    def synchronous_thread_worker(func):
        def creator(*args, **kwargs):
            result = func()
            # If _run_load is a generator, exhaust it so its body runs
            if hasattr(result, "__next__"):
                try:
                    while True:
                        next(result)
                except StopIteration:
                    pass
            return mock_worker

        return creator

    with (
        patch(
            "napari_pyvertexmodel._widget.thread_worker",
            synchronous_thread_worker,
        ),
        patch(
            "napari_pyvertexmodel._widget.VertexModelVoronoiFromTimeImage",
            return_value=mock_v_model,
        ),
        patch("napari_pyvertexmodel._widget._add_surface_layer"),
        patch("napari_pyvertexmodel._widget.progress"),
    ):
        widget._image_layer_load()

    assert "data" in captured_label_data
    np.testing.assert_array_equal(captured_label_data["data"], image_data)


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


def test_update_sliders_from_model_with_advanced_params(
    make_napari_viewer, qtbot
):
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


def test_cancel_button_initial_state(make_napari_viewer, qtbot):
    """Test that cancel button exists and is initially disabled."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    assert widget._cancel_button is not None
    assert widget._cancel_button.enabled is False


def test_cancel_button_in_container(make_napari_viewer, qtbot):
    """Test that cancel button is in the widget container."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    assert widget._cancel_button in widget


def test_cancel_model_no_simulation(make_napari_viewer, qtbot):
    """Test that calling _cancel_model when no simulation is running does nothing."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    # Should not raise any error
    widget._cancel_model()

    # State should be unchanged
    assert widget._cancelled is False


def test_cancel_model_sets_cancelled_flag(make_napari_viewer, qtbot):
    """Test that _cancel_model sets the _cancelled flag when a thread ID is present."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    # Simulate a running simulation by setting a (fake) thread ID
    widget._simulation_thread_id = 99999

    with patch("napari_pyvertexmodel._widget.ctypes") as mock_ctypes:
        mock_ctypes.c_ulong = lambda x: x
        mock_ctypes.py_object = lambda x: x
        mock_ctypes.pythonapi = mock_ctypes.pythonapi  # keep reference
        widget._cancel_model()

    assert widget._cancelled is True


def test_run_model_already_running(make_napari_viewer, qtbot, capsys):
    """Test that starting a second run while one is active prints a message."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    mock_model = Mock()
    mock_model.set = Mock()
    mock_model.create_temporary_folder = Mock(return_value="/tmp")
    widget.v_model = mock_model

    # Simulate a worker already running
    widget._worker = Mock()

    widget._run_model()

    captured = capsys.readouterr()
    assert "already running" in captured.out


def test_on_simulation_finished_resets_state(make_napari_viewer, qtbot):
    """Test that _on_simulation_finished restores button states and closes progress bar."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    # Simulate in-progress state
    widget._run_button.enabled = False
    widget._cancel_button.enabled = True
    widget._worker = Mock()
    widget._simulation_thread_id = 12345
    mock_progress_bar = Mock()
    widget._progress_bar = mock_progress_bar

    widget._on_simulation_finished()

    assert widget._run_button.enabled is True
    assert widget._cancel_button.enabled is False
    assert widget._worker is None
    assert widget._simulation_thread_id is None
    mock_progress_bar.close.assert_called_once()
    assert widget._progress_bar is None


def test_on_simulation_error_prints_message(make_napari_viewer, qtbot, capsys):
    """Test that _on_simulation_error prints an error message."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._on_simulation_error(RuntimeError("test error"))

    captured = capsys.readouterr()
    assert "test error" in captured.out


def test_on_simulation_done_skips_layer_when_cancelled(
    make_napari_viewer, qtbot
):
    """Test that _on_simulation_done does not update layers when cancelled."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._cancelled = True
    widget.v_model = Mock()

    with patch("napari_pyvertexmodel._widget._add_surface_layer") as mock_add:
        widget._on_simulation_done()

    mock_add.assert_not_called()


def test_on_simulation_done_when_not_cancelled(make_napari_viewer, qtbot):
    """Test that _on_simulation_done calls _add_surface_layer when not cancelled."""
    from unittest.mock import Mock

    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._cancelled = False
    widget.v_model = Mock()

    with patch("napari_pyvertexmodel._widget._add_surface_layer") as mock_add:
        widget._on_simulation_done()

    mock_add.assert_called_once_with(
        viewer,
        widget.v_model,
        input_image_dims=None,
    )


def test_run_model_starts_worker(make_napari_viewer, qtbot):
    """Test that _run_model sets up and starts a background thread worker."""
    from unittest.mock import Mock

    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    mock_model = Mock()
    mock_model.set = Mock()
    mock_model.create_temporary_folder = Mock(return_value="/tmp/mock")
    widget.v_model = mock_model

    mock_worker = MagicMock()

    # Replace thread_worker with a version that executes the wrapped function
    # immediately (to cover the inner function body) and returns mock_worker.
    def synchronous_thread_worker(func):
        def creator(*args, **kwargs):
            func()  # execute _run_simulation to cover its lines
            return mock_worker

        return creator

    with (
        patch(
            "napari_pyvertexmodel._widget.thread_worker",
            synchronous_thread_worker,
        ),
        patch("napari_pyvertexmodel._widget.progress"),
    ):
        widget._run_model()

    # Run button should be disabled, cancel button enabled
    assert widget._run_button.enabled is False
    assert widget._cancel_button.enabled is True
    # Worker should be stored and started
    assert widget._worker is mock_worker
    mock_worker.returned.connect.assert_called_once_with(
        widget._on_simulation_done
    )
    mock_worker.errored.connect.assert_called_once_with(
        widget._on_simulation_error
    )
    mock_worker.finished.connect.assert_called_once_with(
        widget._on_simulation_finished
    )
    mock_worker.start.assert_called_once()


def test_progress_bar_initial_state(make_napari_viewer, qtbot):
    """Test that _progress_bar, _load_progress_bar and _load_worker are None on initialization."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    assert widget._progress_bar is None
    assert widget._load_progress_bar is None
    assert widget._load_worker is None


def test_image_layer_load_starts_worker(make_napari_viewer, qtbot):
    """Test that _image_layer_load starts a background worker and updates button states."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    image_data = np.random.rand(50, 50).astype(np.float32)
    image_layer = viewer.add_image(image_data, name="test_image")

    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    widget._image_layer_combo.value = image_layer

    mock_worker = MagicMock()

    def synchronous_thread_worker(func):
        def creator(*args, **kwargs):
            return mock_worker

        return creator

    with (
        patch(
            "napari_pyvertexmodel._widget.thread_worker",
            synchronous_thread_worker,
        ),
        patch("napari_pyvertexmodel._widget.progress"),
    ):
        widget._image_layer_load()

    assert widget._image_layer_load_button.enabled is False
    assert widget._cancel_button.enabled is True
    assert widget._load_worker is mock_worker
    mock_worker.yielded.connect.assert_called_once()
    mock_worker.returned.connect.assert_called_once_with(widget._on_load_done)
    mock_worker.errored.connect.assert_called_once_with(widget._on_load_error)
    mock_worker.finished.connect.assert_called_once_with(
        widget._on_load_finished
    )
    mock_worker.start.assert_called_once()


def test_image_layer_load_already_running(make_napari_viewer, qtbot, capsys):
    """Test that calling _image_layer_load while a load is running prints a message."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    image_data = np.random.rand(50, 50).astype(np.float32)
    image_layer = viewer.add_image(image_data, name="test_image")

    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)
    widget._image_layer_combo.value = image_layer
    widget._load_worker = Mock()  # Simulate an already-running load

    widget._image_layer_load()

    captured = capsys.readouterr()
    assert "already running" in captured.out


def test_on_load_done_updates_model(make_napari_viewer, qtbot):
    """Test that _on_load_done sets v_model and calls _add_surface_layer."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    mock_model = MagicMock()
    label_data = np.zeros((10, 10), dtype=np.uint8)

    with patch("napari_pyvertexmodel._widget._add_surface_layer") as mock_add:
        widget._on_load_done((mock_model, label_data))

    assert widget.v_model is mock_model
    mock_add.assert_called_once()


def test_on_load_done_when_cancelled(make_napari_viewer, qtbot):
    """Test that _on_load_done skips everything when the operation was cancelled."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._cancelled = True
    mock_model = MagicMock()
    label_data = np.zeros((10, 10), dtype=np.uint8)

    with patch("napari_pyvertexmodel._widget._add_surface_layer") as mock_add:
        widget._on_load_done((mock_model, label_data))

    assert widget.v_model is None  # Should not have been updated
    mock_add.assert_not_called()


def test_on_load_done_when_result_is_none(make_napari_viewer, qtbot):
    """Test that _on_load_done handles a (None, None) result (cancelled in thread)."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    with patch("napari_pyvertexmodel._widget._add_surface_layer") as mock_add:
        widget._on_load_done((None, None))

    assert widget.v_model is None
    mock_add.assert_not_called()


def test_on_load_error_prints_message(make_napari_viewer, qtbot, capsys):
    """Test that _on_load_error prints an error message."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._on_load_error(RuntimeError("load failed"))

    captured = capsys.readouterr()
    assert "load failed" in captured.out


def test_on_load_finished_resets_state(make_napari_viewer, qtbot):
    """Test that _on_load_finished restores button states, clears the worker, and closes the progress bar."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    # Simulate in-progress state
    widget._image_layer_load_button.enabled = False
    widget._cancel_button.enabled = True
    widget._load_worker = Mock()
    widget._simulation_thread_id = 99999
    mock_load_progress_bar = Mock()
    widget._load_progress_bar = mock_load_progress_bar

    widget._on_load_finished()

    assert widget._image_layer_load_button.enabled is True
    assert widget._cancel_button.enabled is False
    assert widget._load_worker is None
    assert widget._simulation_thread_id is None
    mock_load_progress_bar.close.assert_called_once()
    assert widget._load_progress_bar is None


def test_on_simulation_finished_no_progress_bar(make_napari_viewer, qtbot):
    """Test _on_simulation_finished works correctly when no progress bar exists."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    widget._progress_bar = None  # Explicitly no progress bar
    # Should not raise
    widget._on_simulation_finished()
    assert widget._progress_bar is None


def test_cancel_cancels_load_labels(make_napari_viewer, qtbot):
    """Test that _cancel_model also cancels a running load-labels operation."""
    from napari_pyvertexmodel._widget import Run3dVertexModel

    viewer = make_napari_viewer
    widget = Run3dVertexModel(viewer)
    qtbot.addWidget(widget.native)

    # Simulate a running load by setting a fake thread ID
    widget._simulation_thread_id = 88888

    with patch("napari_pyvertexmodel._widget.ctypes") as mock_ctypes:
        mock_ctypes.c_ulong = lambda x: x
        mock_ctypes.py_object = lambda x: x
        widget._cancel_model()

    assert widget._cancelled is True
