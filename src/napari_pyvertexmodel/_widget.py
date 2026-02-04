import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import CheckBox, Container, PushButton, create_widget
from napari_pyvertexmodel.utils import _add_surface_layer
from pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import (
    VertexModelVoronoiFromTimeImage,
)
from pyVertexModel.util.utils import load_state

if TYPE_CHECKING:
    import napari

# Default simulation option for pyVertexModel
DEFAULT_VERTEX_MODEL_OPTION = 'wing_disc_equilibrium'
PROJECT_DIRECTORY = Path(__file__).parent.parent.resolve()

# magicgui `Container`
class Run3dVertexModel(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._viewer.dims.ndisplay = 3  # Set viewer to 3D display

        # Image layer selection
        self._image_layer_combo = create_widget(
            label="Input Labels", annotation="napari.layers.Image"
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
        # Lambda Volume slider
        self._lambda_volume_slider = create_widget(
            label=r'$\lambda_V$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._lambda_volume_slider.min = 0
        self._lambda_volume_slider.max = 10
        self._lambda_volume_slider.step = 0.01
        self._lambda_volume_slider.value = 1.0

        # Volume reference slider
        self._volume_reference_slider = create_widget(
            label=r'$V_{0}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._volume_reference_slider.min = 0
        self._volume_reference_slider.max = 10
        self._volume_reference_slider.step = 0.01
        self._volume_reference_slider.value = 1.0

        # Lambda Surface top slider
        self._lambda_surface_top_slider = create_widget(
            label=r'$\lambda_{S1}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._lambda_surface_top_slider.min = 0
        self._lambda_surface_top_slider.max = 10
        self._lambda_surface_top_slider.step = 0.01
        self._lambda_surface_top_slider.value = 0.5

        # Lambda Surface bottom slider
        self._lambda_surface_bottom_slider = create_widget(
            label=r'$\lambda_{S3}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._lambda_surface_bottom_slider.min = 0
        self._lambda_surface_bottom_slider.max = 10
        self._lambda_surface_bottom_slider.step = 0.01
        self._lambda_surface_bottom_slider.value = 0.5

        # Lambda Surface lateral slider
        self._lambda_surface_lateral_slider = create_widget(
            label=r'$\lambda_{S2}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._lambda_surface_lateral_slider.min = 0
        self._lambda_surface_lateral_slider.max = 10
        self._lambda_surface_lateral_slider.step = 0.01
        self._lambda_surface_lateral_slider.value = 0.1

        # Surface Area reference slider
        self._surface_area_reference_slider = create_widget(
            label=r'$A_{0}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._surface_area_reference_slider.min = 0
        self._surface_area_reference_slider.max = 10
        self._surface_area_reference_slider.step = 0.01
        self._surface_area_reference_slider.value = 0.92

        # K substrate slider
        self._k_substrate_slider = create_widget(
            label=r'$k_{Substrate}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._k_substrate_slider.min = 0
        self._k_substrate_slider.max = 1
        self._k_substrate_slider.step = 0.001
        self._k_substrate_slider.value = 0.1

        # T end slider
        self._t_end_slider = create_widget(
            label=r'$t_{end}$',
            annotation=float,
            widget_type="FloatSlider"
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

        # Energy Barrier (Lambda R) slider
        self._lambda_r_slider = create_widget(
            label=r'$\lambda_{R}$',
            annotation=float,
            widget_type="FloatSlider"
        )
        self._lambda_r_slider.min = 0
        self._lambda_r_slider.max = 1e-4
        self._lambda_r_slider.step = 1e-8
        self._lambda_r_slider.value = 8e-7

        # Viscosity slider
        self._viscosity_slider = create_widget(
            label=r'$\nu$',
            annotation=float,
            widget_type="FloatSlider"
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
        self._ablation_checkbox = CheckBox(
            text="Enable Ablation", value=False
        )

        # Cells to ablate slider
        self._cells_to_ablate_slider = create_widget(
            label=r'Cells to Ablate',
            annotation=int,
            widget_type="IntSlider"
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

        # connect your own callbacks
        self._run_button.clicked.connect(self._run_model)
        self._load_simulation_button.clicked.connect(self._load_simulation)
        self._image_layer_load_button.clicked.connect(self._image_layer_load)
        self._show_advanced_params_checkbox.clicked.connect(self._display_advanced_params)

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
                self._surface_area_reference_slider,
                self._k_substrate_slider,
                self._t_end_slider,
                self._show_advanced_params_checkbox,
                self._load_simulation_input,
                self._load_simulation_button,
                self._run_button,
            ]
        )

        # Extended attributes
        self.v_model = None
        self._temp_dir = None  # Store temp directory reference for clean-up
    
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
            if pkl_file is None:
                return

            pkl_file = str(Path(pkl_file))

            if not pkl_file.endswith('.pkl'):
                print("Please select a valid .pkl file.")
                return

            # Load the Vertex Model from the specified file
            self.v_model = VertexModelVoronoiFromTimeImage(
                create_output_folder=False,
                set_option=DEFAULT_VERTEX_MODEL_OPTION,
            )
            load_state(self.v_model, pkl_file)

            self.v_model.set.OutputFolder = None  # Disable output folder
            self.v_model.set.export_images = False  # Disable image export

            # Update sliders with loaded model parameters
            self._update_sliders_from_model()

            print("Simulation loaded successfully.")
            # Save image to viewer
            _add_surface_layer(self._viewer, self.v_model)
        except Exception as e:  # noqa: BLE001
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
            self._cells_to_ablate_slider.value = len(self.v_model.geo.cells_to_ablate)

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
            self.v_model.set.CellsToAblate = np.arange(self._cells_to_ablate_slider.value)

    def _image_layer_load(self):
        try:
            # Load the image layer from the viewer
            image_layer = self._image_layer_combo.value
            if image_layer is None:
                print("Error: No labels layer selected.")
                return

            # Get the label data from the selected layer
            label_data = image_layer.data

            # Create Vertex Model with default parameters
            # Note: This should be updated to use label_data once the
            # VertexModelVoronoiFromTimeImage class supports it
            self.v_model = VertexModelVoronoiFromTimeImage(
                create_output_folder=False,
                set_option=DEFAULT_VERTEX_MODEL_OPTION
            )

            # Set model name and temporary folder
            self.v_model.set.model_name = image_layer.name
            print(f"Loading labels from layer: {image_layer.name}")
            self.v_model.set.OutputFolder = None
            self.v_model.set.export_images = False  # Disable image export
            tempdir = self.v_model.create_temporary_folder()
            self.v_model.set.initial_filename_state = os.path.join(tempdir, image_layer.name)

            # Set number of cells and tissue height
            self.v_model.set.TotalCells = self._tissue_number_of_cells_slider.value
            self.v_model.set.CellHeight = self._tissue_height_slider.value

            # Initialize model
            self.v_model.initialize(label_data)
            print("Labels loaded successfully.")
        except Exception as e:  # noqa: BLE001
            print(f"An error occurred while loading the labels: {e}")
            traceback.print_exc()
            return

        try:
            self._input_image_dims = label_data.shape
            # Save image to viewer
            _add_surface_layer(self._viewer, self.v_model, input_image_dims=self._input_image_dims)
            print("Image layer loaded into the model.")

        except Exception as e:  # noqa: BLE001
            print(f"An error occurred while loading the image layer: {e}")
            traceback.print_exc()

    def _create_temp_folder(self):
        # Create new temp directory and store reference
        self.v_model.create_temporary_folder()

    def _run_model(self):
        if self.v_model is None:
            print("Error: No model loaded. Please load a simulation or labels first.")
            return
        
        self.v_model.t = 0  # Reset time

        # Update model parameters from sliders
        self._update_model_from_sliders()

        # Create temporary folder for output
        self._create_temp_folder()

        # Run the simulation
        self.v_model.iterate_over_time()

        # Save image to viewer
        _add_surface_layer(self._viewer, self.v_model, input_image_dims=self._input_image_dims)


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




