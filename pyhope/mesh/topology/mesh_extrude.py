#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
#
# PyHOPE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyHOPE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyHOPE. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from functools import cache
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def MeshExtrude(mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_progress import ProgressBar
    from pyhope.io.io_gmsh import GMSHCELLTYPES
    from pyhope.mesh.mesh_vars import nGeo
    # ------------------------------------------------------

    points: np.ndarray = mesh.points

    # Instantiate the Gmsh cell type mapping
    gmshCellTypes = GMSHCELLTYPES()

    # Check if the mesh is already 3D
    volume_cells  = [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes3D]
    if volume_cells:
        return mesh

    # Check if the mesh contains 2D elements to extrude
    surface_cells = [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D]
    if surface_cells:
        if not mesh_vars.doExtrude:
            hopout.error('Mesh contains suitable surface cells for extrusion but doExtrude=F, exiting...')
    else:
        hopout.error('Mesh contains no suitable surface cells for extrusion, exiting...')

    hopout.info('Extruding surface to volume mesh')

    # Copy original points
    pointl    = cast(list, mesh.points.tolist())
    # elems_old = mesh.cells.copy()
    # cell_sets = getattr(mesh, 'cell_sets', {})

    # Get base key to distinguish between linear and high-order elements
    ho_key = 100 if nGeo == 1 else 200

    nPoints  = len(pointl)
    nFaces   = np.zeros(2, dtype=int)
    match nGeo:
        case 1:
            faceType = ['triangle'  , 'quad'  ]
            faceNum  = [          3 ,       4 ]
        case 2:
            faceType = ['triangle6' , 'quad9' ]
            faceNum  = [          6 ,       9 ]
        case 3:
            faceType = ['triangle10', 'quad16']
            faceNum  = [         10 ,      16 ]
        case 4:
            faceType = ['triangle15', 'quad25']
            faceNum  = [         15 ,      25 ]
        case _:
            hopout.error('nGeo = {} not supported for element splitting'.format(nGeo))

    # Prepare new cell blocks and new cell_sets
    elems_lst = {ftype: [] for ftype in faceType}
    csets_lst = {}

    # Set up the element extrusion function
    elemExtruder = {'quad': (extrude_hex  , hex_faces    )}
    faceMaper = { ho_key + 4: lambda x: 0,
                  ho_key + 5: lambda x: 0 if x == 0 else 1,
                  ho_key + 6: lambda x: 0 if x == 0 else 1,
                  ho_key + 8: lambda x: 1}
    nFace   = (nGeo+1)*(nGeo+2)/2

    # Create the element sets
    meshcells = tuple((k, v) for k, v in mesh.cell_sets_dict.items() if any(key.startswith('line') for key in v.keys())
                                                                     or any(key.startswith('quad') for key in v.keys()))

    # If meshcells is empty, we fake it assign it to Zone1
    if len(meshcells) == 0:
        meshcells = tuple(('Zone1', {k: np.array([i for i in range(len(v))])}) for k, v in mesh.cells_dict.items()
                                                                                        if (k.startswith('tria')
                                                                                        or  k.startswith('quad')))  # noqa: E271

    nTotalElems = sum(cdata.shape[0] for _, zdata in meshcells for _, cdata in cast(dict, zdata).items())
    bar = ProgressBar(value=nTotalElems, title='│             Processing Elements', length=33, threshold=1000)

    for iElem, meshcell in enumerate(meshcells):
        _    , mdict = meshcell
        mtype, mcell = list(cast(dict, mdict).keys())[0], list(cast(dict, mdict).values())[0]

        extrude, faces = elemExtruder.get(mtype, (None, None))
        # FIXME: Use the actual element type to get the faces
        faceMap = faceMaper.get(108, None)

        # Sanity check
        if faceMap is None:
            raise ValueError('Missing faceMap for element type {}'.format(mtype))

        cdata   = mesh.get_cells_type(mtype)[mcell]

        if extrude is None or faces is None:
            hopout.error('Element type {} not supported for extruding'.format(mtype), traceback=True)

        # Face block: Iterate over each element
        for elem in cdata:
            # Generate new nodes
            newNodes, newPoints = extrude(points[cdata], nGeo)
            # Append the new point to the point list
            pointl.extend(newPoints.squeeze().tolist())

            # Overwrite the element with the new indices
            extElem  = elem.tolist() + np.add(newNodes, nPoints).tolist()
            nPoints += len(newNodes)

            # FIXME: Set the correct element type depending on the surface element
            elemName = 'hexahedron'
            # The list is crucial here, we want to build a list of lists
            elems_lst.setdefault(elemName, []).extend([extElem])

            # Create the new faces
            subFaces = tuple(np.array(extElem)[face] for face in faces(nGeo))

            for subFace in subFaces:
                faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)
                # faceSet = frozenset(subFace)

                # Use the associated boundary name
                # (Assuming all boundary names are stored in a list for this candidate. Adjust if needed.)
                # FIXME: Load the actual boundary conditions
                # names = csets_old[candidate]
                names = ['zplus']
                # Update csets_lst for each name in the list.
                for name in names:
                    csets_lst.setdefault(name, [[], []])
                    csets_lst[name][faceVal].append(nFaces[faceVal])

                elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))
                nFaces[faceVal] += 1

            # Update the progress bar
            bar.step()

    # Close the progress bar
    bar.close()

    # Convert lists to NumPy arrays for elems_new and csets_new
    elems_new = {}
    csets_new = {}

    for key in elems_lst:
        if   isinstance(elems_lst[key], list) and     elems_lst[key]:  # noqa: E271
            # Convert the list of accumulated arrays/lists into a single NumPy array
            elems_new[key] = np.array(elems_lst[key], dtype=int)
        elif isinstance(elems_lst[key], list) and not elems_lst[key]:
            # Determine the expected number of columns
            elems_new[key] = np.empty((0, faceNum[faceType.index(key)]), dtype=int)

    for key in csets_lst:
        csets_new[key] = tuple(np.array(lst, dtype=int) for lst in csets_lst[key])

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251

    hopout.sep()
    return mesh


# @cache
def extrude_hex(points: np.ndarray,
                order: int) -> tuple[np.ndarray, ...]:
    match order:
        case 1:
            # Repeat the bottom layer twice
            nodes   = np.arange(4, dtype=np.int64).tolist()
            # FIXME: Use the actual extrusionVector
            extrude = points + np.array([0.0, 0.0, 1.0], dtype=points.dtype)
        # FIXME: Implement the other orders

    return nodes, extrude


@cache
def hex_faces(order: int) -> tuple[np.ndarray, ...]:
    """ Given the indices of a hexahedral element, return a tuple with the 6 faces as arrays
    """
    match order:
        case 1:
            return (np.array((  0,  1,  2,  3), dtype=int),
                    np.array((  0,  1,  5,  4), dtype=int),
                    np.array((  1,  2,  6,  5), dtype=int),
                    np.array((  2,  6,  7,  3), dtype=int),
                    np.array((  0,  4,  7,  3), dtype=int),
                    np.array((  4,  5,  6,  7), dtype=int))
        # INFO: It would be better to return the actual high-order faces here but PyHOPE will automatically fallback to the corner
        #       nodes if the inner nodes are not available
        case _:
            return (np.array((  0,  1,  2,  3), dtype=int),
                    np.array((  0,  1,  5,  4), dtype=int),
                    np.array((  1,  2,  6,  5), dtype=int),
                    np.array((  2,  6,  7,  3), dtype=int),
                    np.array((  0,  4,  7,  3), dtype=int),
                    np.array((  4,  5,  6,  7), dtype=int))
