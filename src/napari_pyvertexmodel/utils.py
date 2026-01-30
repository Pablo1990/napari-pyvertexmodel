from __future__ import annotations

from typing import Any

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy


def _add_surface_layer(viewer, v_model):
    """
    Add surface layer to napari viewer
    :param v_model:
    :return:
    """

    for _cell_id, c_cell in enumerate(v_model.geo.Cells):
        layer_name_cell, t_faces, t_scalars, t_vertices = _create_surface_data(c_cell, v_model)

        try:
            # if the layer exists, update the data
            curr_verts, curr_faces, curr_values = viewer.layers[
                layer_name_cell
            ].data

            viewer.layers[layer_name_cell].data = (
                np.concatenate((curr_verts, t_vertices), axis=0),
                np.concatenate((curr_faces, t_faces), axis=0),
                np.concatenate((curr_values, t_scalars), axis=0),
            )

            # Update timepoint that is displayed
            viewer.dims.set_current_step(0, v_model.t)

        except KeyError:
            # otherwise add it to the viewer
            viewer.add_surface(
                (t_vertices, t_faces, t_scalars),
                colormap="viridis",
                opacity=0.9,
                contrast_limits=[0, 1],
                name=layer_name_cell,
            )


def _create_surface_data(c_cell, v_model) -> tuple[str, Any, Any, Any]:
    """
    Create surface data for a cell
    :param c_cell:
    :param v_model:
    :return: layer_name_cell, t_faces, t_scalars, t_vertices
    """
    layer_name_cell = f"{v_model.set.model_name}_cell_{c_cell.ID}"
    vtk_poly = c_cell.create_vtk()
    t_vertices = vtk_to_numpy(vtk_poly.GetPoints().GetData())
    t_faces = vtk_to_numpy(vtk_poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    t_scalars = vtk_poly.GetCellData().GetScalars()

    return layer_name_cell, t_faces, t_scalars, t_vertices

