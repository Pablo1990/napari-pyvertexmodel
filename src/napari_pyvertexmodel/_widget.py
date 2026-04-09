import copy
import ctypes
import os
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import napari.layers
import numpy as np
from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    PushButton,
    create_widget,
)
from napari.qt.threading import thread_worker
from pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import (
    VertexModelVoronoiFromTimeImage,
)
from pyVertexModel.util.utils import load_state

from napari_pyvertexmodel.utils import _add_surface_layer

if TYPE_CHECKING:
    import napari

# Default simulation option for pyVertexModel
DEFAULT_VERTEX_MODEL_OPTION = "wing_disc_equilibrium"
PROJECT_DIRECTORY = Path(__file__).parent.parent.resolve()


# magicgui `Container`
class Run3dVertexModel(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._viewer.dims.ndisplay = 3  # Set viewer to 3D display

        # Image/Labels layer selection (accepts both Image and Labels layers)
        self._image_layer_combo = create_widget(
            label="Input Labels",
            annotation=napari.layers.Layer,
        )
        # Add number of cells
        self._tissue_number_of_cells_slider = create_widget(
            label="Number of Cells", annotation=int, widget_type="IntSlider"
        )
        self._tissue_number_of_cells_slider.min = 10
        self._tissue_number_of_cells_slider.max = 200
        self._tissue_number_of_cells_slider.step = 1
        self._tissue_number_of_cells_slider.value = 20
        # Add tissue height
        self._tissue_height_slider = create_widget(
            label="Tissue Height", annotation=float, widget_type="FloatSlider"
        )
        self._tissue_height_slider.min = 0.01
        self._tissue_height_slider.max = 100.0
        self._tissue_height_slider.step = 0.01
        self._tissue_height_slider.value = 50

        # Button to load image layer
        self._image_layer_load_button = PushButton(text="Load Labels")

        # ----- Sliders with mechanical parameters -----
        # Lambda Volume slider (use Unicode lambda)
        self._lambda_volume_slider = create_widget(
            label="λV", annotation=float, widget_type="FloatSlider"
        )
        self._lambda_volume_slider.min = 0
        self._lambda_volume_slider.max = 10
        self._lambda_volume_slider.step = 0.01
        self._lambda_volume_slider.value = 1.0

        # Volume reference slider (use subscript zero)
        self._volume_reference_slider = create_widget(
            label="V₀", annotation=float, widget_type="FloatSlider"
        )
        self._volume_reference_slider.min = 0
        self._volume_reference_slider.max = 10
        self._volume_reference_slider.step = 0.01
        self._volume_reference_slider.value = 1.0

        # Lambda Surface top slider (use Unicode lambda)
        self._lambda_surface_top_slider = create_widget(
            label="λS1",
            annotation=float,
            widget_type="FloatSlider",
        )
        self._lambda_surface_top_slider.min = 0
        self._lambda_surface_top_slider.max = 10
        self._lambda_surface_top_slider.step = 0.01
        self._lambda_surface_top_slider.value = 0.5

        # Lambda Surface bottom slider (use Unicode lambda)
        self._lambda_surface_bottom_slider = create_widget(
            label="λS3",
            annotation=float,
            widget_type="FloatSlider",
        )
        self._lambda_surface_bottom_slider.min = 0
        self._lambda_surface_bottom_slider.max = 10
        self._lambda_surface_bottom_slider.step = 0.01
        self._lambda_surface_bottom_slider.value = 0.5

        # Lambda Surface lateral slider (use Unicode lambda)
        self._lambda_surface_lateral_slider = create_widget(
            label="λS2",
            annotation=float,
            widget_type="FloatSlider",
        )
        self._lambda_surface_lateral_slider.min = 0
        self._lambda_surface_lateral_slider.max = 10
        self._lambda_surface_lateral_slider.step = 0.01
        self._lambda_surface_lateral_slider.value = 0.1

        # Button to estimate lambda_s parameters (maintains cell aspect ratio)
        self._estimate_lambda_s_button = PushButton(
            text="Estimate \u03bbS Parameters"
        )

        # Surface Area reference slider (use subscript zero)
        self._surface_area_reference_slider = create_widget(
            label="A₀", annotation=float, widget_type="FloatSlider"
        )
        self._surface_area_reference_slider.min = 0
        self._surface_area_reference_slider.max = 10
        self._surface_area_reference_slider.step = 0.01
        self._surface_area_reference_slider.value = 0.92

        # K substrate slider
        self._k_substrate_slider = create_widget(
            label="Substrate adhesion (k)",
            annotation=float,
            widget_type="FloatSlider",
        )
        self._k_substrate_slider.min = 0
        self._k_substrate_slider.max = 1
        self._k_substrate_slider.step = 0.001
        self._k_substrate_slider.value = 0.1

        # T end slider
        self._t_end_slider = create_widget(
            label="End time (min.)", annotation=float, widget_type="FloatSlider"
        )
        self._t_end_slider.min = 0
        self._t_end_slider.max = 30
        self._t_end_slider.step = 0.1
        self._t_end_slider.value = 1

        # --- Advanced mechanical parameters hidden by default ---
        # Checkbox to show/hide advanced parameters
        self._show_advanced_params_checkbox = CheckBox(
            text="Show Advanced Parameters", value=False
        )

        # Energy Barrier (Lambda AR) slider (use Unicode lambda)
        self._lambda_r_slider = create_widget(
            label="λ Aspect Ratio", annotation=float, widget_type="FloatSlider"
        )
        self._lambda_r_slider.min = 0
        self._lambda_r_slider.max = 1e-4
        self._lambda_r_slider.step = 1e-8
        self._lambda_r_slider.value = 8e-7

        # Viscosity slider (use Unicode nu)
        self._viscosity_slider = create_widget(
            label="Viscosity (µ)", annotation=float, widget_type="FloatSlider"
        )
        self._viscosity_slider.min = 0
        self._viscosity_slider.max = 1
        self._viscosity_slider.step = 0.001
        self._viscosity_slider.value = 0.07

        # Remodelling checkbox
        self._remodelling_checkbox = CheckBox(
            text="Enable Remodelling", value=False
        )

        # Ablation checkbox
        self._ablation_checkbox = CheckBox(text="Enable Ablation", value=False)

        # Cells to ablate slider
        self._cells_to_ablate_slider = create_widget(
            label=r"Cells to Ablate", annotation=int, widget_type="IntSlider"
        )
        self._cells_to_ablate_slider.min = 0
        self._cells_to_ablate_slider.max = 10
        self._cells_to_ablate_slider.step = 1
        self._cells_to_ablate_slider.value = 0

        # -----------------------------------------------
        # Load simulation input
        self._load_simulation_input = create_widget(
            label="Load Simulation", annotation=str, widget_type="FileEdit"
        )
        self._load_simulation_button = PushButton(text="Load")

        # Add button to run Vertex Model
        self._run_button = PushButton(text="Run it!")

        # Add button to cancel a running simulation (disabled until simulation starts)
        self._cancel_button = PushButton(text="Cancel")
        self._cancel_button.enabled = False

        # Status Display
        self.status_label = Label(value="Ready")
        self.progress_label = Label(value="")

        # connect your own callbacks
        self._run_button.clicked.connect(self._run_model)
        self._cancel_button.clicked.connect(self._cancel_model)
        self._load_simulation_button.clicked.connect(self._load_simulation)
        self._image_layer_load_button.clicked.connect(self._image_layer_load)
        self._show_advanced_params_checkbox.clicked.connect(
            self._display_advanced_params
        )
        self._estimate_lambda_s_button.clicked.connect(self._estimate_lambda_s)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._image_layer_combo,
                self._tissue_number_of_cells_slider,
                self._tissue_height_slider,
                self._image_layer_load_button,
                self._lambda_volume_slider,
                self._volume_reference_slider,
                self._lambda_surface_top_slider,
                self._lambda_surface_bottom_slider,
                self._lambda_surface_lateral_slider,
                self._estimate_lambda_s_button,
                self._surface_area_reference_slider,
                self._k_substrate_slider,
                self._t_end_slider,
                self._show_advanced_params_checkbox,
                self._load_simulation_input,
                self._load_simulation_button,
                self._run_button,
                self._cancel_button,
                self.status_label,
                self.progress_label,
            ]
        )

        # Extended attributes
        self.v_model = None
        self._temp_dir = None  # Store temp directory reference for clean-up
        self._worker = None  # Background simulation worker
        self._load_worker = None  # Background load-labels worker
        self._estimate_lambda_s_worker = None  # Background estimation worker
        self._simulation_thread_id = None  # Thread ID for cancellation
        self._simulation_lock = (
            threading.Lock()
        )  # Protects _simulation_thread_id
        self._cancelled = False  # Whether cancellation was requested

    def __del__(self):
        """Clean-up temporary directory on widget destruction."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:  # noqa: BLE001
                pass  # Silently ignore cleanup errors during destruction

    def _load_simulation(self):
        try:
            pkl_file = self._load_simulation_input.value
            if pkl_file is None or pkl_file == "":
                return

            pkl_file = str(Path(pkl_file))

            if not pkl_file.endswith(".pkl"):
                print("Please select a valid .pkl file.")
                return

            self.status_label.value = "Loading simulation..."
            self.progress_label.value = ""
            self.v_model = VertexModelVoronoiFromTimeImage(
                create_output_folder=False,
                set_option=DEFAULT_VERTEX_MODEL_OPTION,
            )

            self.status_label.value = "Loading simulation state..."
            load_state(self.v_model, pkl_file)
            self.v_model.set.OutputFolder = None  # Disable output folder
            self.v_model.set.export_images = False  # Disable image export
            self._update_sliders_from_model()

            self.status_label.value = "Updating viewer..."
            _add_surface_layer(self._viewer, self.v_model)
            self.status_label.value = "Ready"
            self.progress_label.value = "Simulation loaded successfully."
            print("Simulation loaded successfully.")
        except Exception as e:  # noqa: BLE001
            self.status_label.value = "Ready"
            self.progress_label.value = f"Error loading simulation: {e}"
            print(f"An error occurred while loading the simulation: {e}")

    def _update_sliders_from_model(self):
        self._tissue_number_of_cells_slider.value = self.v_model.set.TotalCells
        self._tissue_height_slider.value = self.v_model.set.CellHeight
        self._lambda_volume_slider.value = self.v_model.set.lambdaV
        self._lambda_surface_top_slider.value = self.v_model.set.lambdaS1
        self._lambda_surface_bottom_slider.value = self.v_model.set.lambdaS3
        self._lambda_surface_lateral_slider.value = self.v_model.set.lambdaS2
        self._volume_reference_slider.value = self.v_model.set.ref_V0
        self._surface_area_reference_slider.value = self.v_model.set.ref_A0
        self._k_substrate_slider.value = self.v_model.set.kSubstrate
        self._t_end_slider.value = self.v_model.set.tend

        if self._show_advanced_params_checkbox.value:
            self._lambda_r_slider.value = self.v_model.set.lambdaR
            self._viscosity_slider.value = self.v_model.set.nu
            self._remodelling_checkbox.value = self.v_model.set.RemodelCells
            self._ablation_checkbox.value = self.v_model.set.AblateCells
            self._cells_to_ablate_slider.value = len(
                self.v_model.geo.cells_to_ablate
            )

    def _update_model_from_sliders(self):
        self.v_model.set.lambdaV = self._lambda_volume_slider.value
        self.v_model.set.lambdaS1 = self._lambda_surface_top_slider.value
        self.v_model.set.lambdaS3 = self._lambda_surface_bottom_slider.value
        self.v_model.set.lambdaS2 = self._lambda_surface_lateral_slider.value
        self.v_model.set.ref_V0 = self._volume_reference_slider.value
        self.v_model.set.ref_A0 = self._surface_area_reference_slider.value
        self.v_model.set.kSubstrate = self._k_substrate_slider.value
        self.v_model.set.tend = self._t_end_slider.value

        if self._show_advanced_params_checkbox.value:
            self.v_model.set.lambdaR = self._lambda_r_slider.value
            self.v_model.set.nu = self._viscosity_slider.value
            self.v_model.set.RemodelCells = self._remodelling_checkbox.value
            self.v_model.set.AblateCells = self._ablation_checkbox.value
            self.v_model.set.CellsToAblate = np.arange(
                self._cells_to_ablate_slider.value
            )

    def _image_layer_load(self):
        """Start loading labels in a background thread (cancellable)."""
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            print("Error: No labels layer selected.")
            return

        # Get the label data from the selected layer.
        # For a Labels layer, convert to a binary image (non-zero → 1)
        # so that the simulation receives a standard segmented image.
        if isinstance(image_layer, napari.layers.Labels):
            label_data = image_layer.data == 0
        elif isinstance(image_layer, napari.layers.Image):
            # Check if the image has a 0 background or background of 1
            label_data = image_layer.data
        else:
            print("Error: Selected layer must be an Image or Labels layer.")
            return

        if self._load_worker is not None:
            print("Load labels already running.")
            return

        # Snapshot mutable slider values before entering the thread
        layer_name = image_layer.name
        total_cells = self._tissue_number_of_cells_slider.value
        cell_height = self._tissue_height_slider.value

        self._cancelled = False
        self._simulation_thread_id = None
        self._image_layer_load_button.enabled = False
        self._cancel_button.enabled = True
        self.status_label.value = "Loading labels..."
        self.progress_label.value = ""

        @thread_worker
        def _run_load():
            with self._simulation_lock:
                self._simulation_thread_id = threading.current_thread().ident
            try:
                local_model = VertexModelVoronoiFromTimeImage(
                    create_output_folder=False,
                    set_option=DEFAULT_VERTEX_MODEL_OPTION,
                )
                local_model.set.model_name = layer_name
                local_model.set.OutputFolder = None
                local_model.set.export_images = False
                tempdir = local_model.create_temporary_folder()
                local_model.set.initial_filename_state = os.path.join(
                    tempdir, layer_name
                )
                local_model.set.TotalCells = total_cells
                local_model.set.CellHeight = cell_height
                local_model.initialize(label_data)
            except SystemExit:
                print("Load labels cancelled.")
                return None, None

            return local_model, label_data

        worker = _run_load()
        worker.returned.connect(self._on_load_done)
        worker.errored.connect(self._on_load_error)
        worker.finished.connect(self._on_load_finished)
        self._load_worker = worker
        worker.start()

    def _on_load_done(self, result):
        """Called on the main thread when labels loading completes."""
        local_model, label_data = result
        if local_model is None or self._cancelled:
            return

        self.v_model = local_model
        self.progress_label.value = "Labels loaded successfully."
        print("Labels loaded successfully.")

        try:
            self._input_image_dims = label_data.shape
            _add_surface_layer(
                self._viewer,
                self.v_model,
                input_image_dims=self._input_image_dims,
            )
            print("Image layer loaded into the model.")
        except Exception as e:  # noqa: BLE001
            print(f"An error occurred while loading the image layer: {e}")
            traceback.print_exc()

    def _on_load_error(self, exc):
        """Called on the main thread when labels loading raises an unhandled exception."""
        self.progress_label.value = f"Error loading labels: {exc}"
        print(f"An error occurred while loading the labels: {exc}")
        traceback.print_exc()

    def _on_load_finished(self):
        """Called on the main thread when the load worker finishes (success or error)."""
        self._image_layer_load_button.enabled = True
        self._cancel_button.enabled = False
        self._load_worker = None
        self._simulation_thread_id = None
        self.status_label.value = "Ready"

    def _create_temp_folder(self):
        # Create new temp directory and store reference
        self.v_model.create_temporary_folder()

    def _run_model(self):
        if self.v_model is None:
            print(
                "Error: No model loaded. Please load a simulation or labels first."
            )
            return

        if self._worker is not None:
            print("Simulation already running.")
            return

        self.v_model.t = 0  # Reset time

        # Update model parameters from sliders
        self._update_model_from_sliders()

        # Create temporary folder for output
        self._create_temp_folder()

        self._cancelled = False
        self._simulation_thread_id = None
        self._run_button.enabled = False
        self._cancel_button.enabled = True
        self.status_label.value = "Running simulation..."
        self.progress_label.value = ""

        @thread_worker
        def _run_simulation():
            with self._simulation_lock:
                self._simulation_thread_id = threading.current_thread().ident
            try:
                self.v_model.iterate_over_time()
            except SystemExit:
                print("Simulation cancelled.")

        worker = _run_simulation()
        worker.returned.connect(self._on_simulation_done)
        worker.errored.connect(self._on_simulation_error)
        worker.finished.connect(self._on_simulation_finished)
        self._worker = worker
        worker.start()

    def _on_simulation_done(self, result=None):
        """Called on the main thread when the simulation completes successfully."""
        if not self._cancelled:
            _add_surface_layer(
                self._viewer,
                self.v_model,
                input_image_dims=getattr(self, "_input_image_dims", None),
            )
            self.progress_label.value = "Simulation complete."

    def _on_simulation_error(self, exc):
        """Called on the main thread when the simulation raises an unhandled exception."""
        self.progress_label.value = f"Error: {exc}"
        print(f"An error occurred during the simulation: {exc}")

    def _on_simulation_finished(self):
        """Called on the main thread when the simulation worker finishes (success or error)."""
        self._run_button.enabled = True
        self._cancel_button.enabled = False
        self._worker = None
        self._simulation_thread_id = None
        self.status_label.value = "Ready"

    def _cancel_model(self):
        """Request cancellation of the currently running simulation.

        Uses ``ctypes.PyThreadState_SetAsyncExc`` to inject a ``SystemExit``
        exception into the simulation thread, causing ``iterate_over_time()``
        to stop at the next safe point. This is the most practical approach
        given that pyVertexModel does not expose a cooperative cancellation API.
        """
        with self._simulation_lock:
            thread_id = self._simulation_thread_id
        if thread_id is not None:
            self._cancelled = True
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread_id),
                ctypes.py_object(SystemExit),
            )
            print("Cancellation requested. The simulation will stop shortly.")
            self.progress_label.value = "Cancellation requested..."

    def _display_advanced_params(self):
        if self._show_advanced_params_checkbox.value:
            # Show advanced parameters
            for widget in [
                self._lambda_r_slider,
                self._viscosity_slider,
                self._remodelling_checkbox,
                self._ablation_checkbox,
                self._cells_to_ablate_slider,
            ]:
                if widget not in self:
                    self.append(widget)
        else:
            # Hide advanced parameters
            for widget in [
                self._lambda_r_slider,
                self._viscosity_slider,
                self._remodelling_checkbox,
                self._ablation_checkbox,
                self._cells_to_ablate_slider,
            ]:
                if widget in self:
                    self.remove(widget)

    def _estimate_lambda_s(self):
        """Estimate lambdaS1 and lambdaS2 to maintain the cell aspect ratio.

        Uses optuna to minimise the gradient (gr) from a single model
        iteration, keeping kSubstrate=0, EnergyBarrierAR=False, lambdaR=0,
        ref_A0=0 and ref_V0=1 so that only volume and surface-area terms
        contribute.  lambdaS3 is constrained to equal lambdaS1 throughout.
        The best parameters found are written back to the λS sliders.
        """
        if self.v_model is None:
            print(
                "Error: No model loaded. "
                "Please load a simulation or labels first."
            )
            return

        if self._estimate_lambda_s_worker is not None:
            print("Estimation already running.")
            return

        self.status_label.value = "Estimating \u03bbS parameters..."
        self.progress_label.value = ""
        self._estimate_lambda_s_button.enabled = False

        # Snapshot the current model so the background thread works on a copy
        v_model_snapshot = copy.deepcopy(self.v_model)

        @thread_worker
        def _run_estimation():
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                local_model = copy.deepcopy(v_model_snapshot)
                local_model.set.lambdaS1 = trial.suggest_float(
                    "lambdaS1", 1e-7, 1
                )
                local_model.set.lambdaS2 = trial.suggest_float(
                    "lambdaS2", 1e-7, 1
                )
                local_model.set.lambdaS3 = local_model.set.lambdaS1
                local_model.set.kSubstrate = 0
                local_model.set.EnergyBarrierAR = False
                local_model.set.lambdaR = 0
                local_model.set.ref_A0 = 0
                local_model.set.ref_V0 = 1
                local_model.geo.init_reference_values_and_noise(
                    local_model.set
                )
                try:
                    gr = local_model.single_iteration(post_operations=False)
                    return gr
                except Exception:  # noqa: BLE001
                    return float("inf")

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)
            return study.best_params

        worker = _run_estimation()
        worker.returned.connect(self._on_estimation_done)
        worker.errored.connect(self._on_estimation_error)
        worker.finished.connect(self._on_estimation_finished)
        self._estimate_lambda_s_worker = worker
        worker.start()

    def _on_estimation_done(self, best_params):
        """Called on the main thread when the estimation completes."""
        if best_params is None:
            return
        lambda_s1 = best_params.get(
            "lambdaS1", self._lambda_surface_top_slider.value
        )
        lambda_s2 = best_params.get(
            "lambdaS2", self._lambda_surface_lateral_slider.value
        )
        self._lambda_surface_top_slider.value = lambda_s1
        self._lambda_surface_bottom_slider.value = lambda_s1  # S3 = S1
        self._lambda_surface_lateral_slider.value = lambda_s2
        self.progress_label.value = (
            f"\u03bbS1=\u03bbS3={lambda_s1:.4f}, " f"\u03bbS2={lambda_s2:.4f}"
        )
        print(
            f"Best \u03bbS parameters: "
            f"\u03bbS1=\u03bbS3={lambda_s1:.4f}, \u03bbS2={lambda_s2:.4f}"
        )

    def _on_estimation_error(self, exc):
        """Called on the main thread when the estimation raises an exception."""
        self.progress_label.value = f"Error estimating \u03bbS: {exc}"
        print(f"An error occurred while estimating \u03bbS parameters: {exc}")

    def _on_estimation_finished(self):
        """Called on the main thread when the estimation worker finishes."""
        self._estimate_lambda_s_button.enabled = True
        self.status_label.value = "Ready"
        self._estimate_lambda_s_worker = None
