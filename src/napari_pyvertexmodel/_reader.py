"""
This module implements a reader plugin for napari that can read
both .npy and .pkl files.

- .npy files: Numpy arrays (currently limited to integer arrays)
- .pkl files: pyVertexModel simulation states that are loaded and
  visualized as surface layers in napari

It implements the Reader specification for napari.
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
from pathlib import Path

import numpy as np

from napari_pyvertexmodel.utils import _create_surface_data

# Default simulation option for pyVertexModel
DEFAULT_VERTEX_MODEL_OPTION = 'wing_disc_equilibrium'


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # Check if the file is a .pkl or .npy file
    path_obj = Path(path)
    if path_obj.suffix.lower() == '.pkl':
        return pkl_reader_function
    elif path_obj.suffix.lower() == '.npy':
        # Check if it's a valid numpy array
        try:
            arr = np.load(path, mmap_mode='r')
            if arr.dtype != np.int_:
                return None
        except OSError:
            return None
        return npy_reader_function

    return None


def pkl_reader_function(path):
    """Read a .pkl file and return layer data.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple contains
        (data, metadata, layer_type).
    """
    from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import (
        VertexModelVoronoiFromTimeImage,
    )
    from src.pyVertexModel.util.utils import load_state

    # Handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    layer_data_list = []

    for file_path in paths:
        try:
            # Load the vertex model state from pickle file
            v_model = VertexModelVoronoiFromTimeImage(
                create_output_folder=False,
                set_option=DEFAULT_VERTEX_MODEL_OPTION
            )
            load_state(v_model, file_path)

            # Convert the vertex model to surface layer data
            for _cell_id, c_cell in enumerate(v_model.geo.Cells):
                if c_cell.AliveStatus is not None:
                    layer_name_cell, t_faces, t_scalars, t_vertices = _create_surface_data(c_cell, v_model)

                    # Create surface layer data
                    surface_data = (t_vertices, t_faces, t_scalars)
                    add_kwargs = {
                        'name': layer_name_cell,
                        'colormap': 'plasma',
                        'opacity': 0.9,
                        'contrast_limits': [0, 1]
                    }
                    layer_type = "surface"

                    layer_data_list.append((surface_data, add_kwargs, layer_type))
        except Exception as e:  # noqa: BLE001
            # Log error but continue with other files
            print(f"Error loading {file_path}: {e}")
            continue

    return layer_data_list


def npy_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
