from __future__ import annotations

from typing import Any

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy


def _add_surface_layer(viewer, v_model):
    """
    Add surface layer to napari viewer.
    Batches all cells into a single layer for better performance.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    v_model : VertexModel
        The vertex model containing cells to visualize
    """
    # Batch all cells into a single layer for better performance
    all_faces, all_scalars, all_vertices = _get_mesh(v_model)

    # Only create/update layer if we have data
    if all_vertices:
        # Concatenate all cell data into single arrays
        merged_vertices = np.concatenate(all_vertices, axis=0)
        merged_faces = np.concatenate(all_faces, axis=0)
        merged_scalars = np.concatenate(all_scalars, axis=0)

        layer_name = f"{v_model.set.model_name}_all_cells"

        try:
            # if the layer exists, update the data
            viewer.layers[layer_name].data = (
                merged_vertices,
                merged_faces,
                merged_scalars,
            )

            # Update timepoint that is displayed
            viewer.dims.set_current_step(0, v_model.t)

        except KeyError:
            # otherwise add it to the viewer
            viewer.add_surface(
                (merged_vertices, merged_faces, merged_scalars),
                colormap="plasma",
                opacity=0.9,
                contrast_limits=[0.0001, 0.0006],
                name=layer_name,
            )


def _get_mesh(v_model) -> tuple[list[Any], list[Any], list[Any]]:
    all_vertices = []
    all_faces = []
    all_scalars = []

    # Accumulate offset for face indices
    offset_indices = 0
    for _cell_id, c_cell in enumerate(v_model.geo.Cells):
        if c_cell.AliveStatus is not None:
            _, t_faces, t_scalars, t_vertices = _create_surface_data(c_cell, v_model, offset_indices=offset_indices)

            all_vertices.append(t_vertices)
            all_faces.append(t_faces)
            all_scalars.append(t_scalars)

            # Update offset for next cell
            offset_indices += t_vertices.shape[0]
    return all_faces, all_scalars, all_vertices


def _create_surface_data(c_cell, v_model, offset_indices=0) -> tuple[str, Any, Any, Any]:
    """
    Create surface data for a cell
    :param c_cell:
    :param v_model:
    :return: layer_name_cell, t_faces, t_scalars, t_vertices
    """
    # Create VTK polydata for the cell
    layer_name_cell = f"{v_model.set.model_name}_cell_{c_cell.ID}"
    vtk_poly = c_cell.create_vtk()
    t_vertices = vtk_to_numpy(vtk_poly.GetPoints().GetData())
    t_faces = vtk_to_numpy(vtk_poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    t_scalars = vtk_to_numpy(vtk_poly.GetCellData().GetScalars())

    # Check t_scalars length matches number of vertices
    if len(t_scalars) != t_vertices.shape[0]:
        if len(t_scalars) > t_vertices.shape[0]:
            t_scalars = t_scalars[: t_vertices.shape[0]]
        else:
            # If there are fewer scalars than vertices, pad with zeros
            padding = np.zeros(t_vertices.shape[0] - len(t_scalars))
            t_scalars = np.concatenate((t_scalars, padding))

    # Adjust face indices with offset integer
    t_faces += offset_indices

    # Return the layer name and the surface data
    return layer_name_cell, t_faces, t_scalars, t_vertices

