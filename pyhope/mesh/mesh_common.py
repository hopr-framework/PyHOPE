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
import sys
from functools import cache
from typing import Union
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


@cache
def faces(elemType: Union[int, str]) -> list[str]:
    """ Return a list of all sides of a hexahedron
    """
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: ['z-', 'y-', 'x+', 'y+', 'x-', 'z+']
                }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in faces: elemType {elemType} is not supported')

    return faces_map[elemType % 100]


@cache
def edge_to_dir(edge: int, elemType: Union[int, str]) -> int:
    """ GMSH: Create edges from points in the given direction
    """
    dir_map  = {  # Tetrahedron
                  # Pyramid
                  # Wedge / Prism
                  # Hexahedron
                  8: {  0:  0,  2:  0,  4:  0,  6:  0,  # Direction 0
                        1:  1,  3:  1,  5:  1,  7:  1,  # Direction 1
                        8:  2,  9:  2, 10:  2, 11:  2}  # Direction 2
               }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in dir_map:
        raise ValueError(f'Error in edge_to_direction: elemType {elemType} is not supported')

    dir = dir_map[elemType % 100]

    try:
        return dir[edge]
    except KeyError:
        raise KeyError(f'Error in edge_to_dir: edge {edge} is not supported')


@cache
def edge_to_corner(edge: int, elemType: Union[int, str], dtype=int) -> np.ndarray:
    """ GMSH: Get points on edges
    """
    edge_map = {  # Tetrahedron
                  4: [ [0, 1], [1, 2], [2, 1], [0, 3],
                       [1, 3], [2, 3]                 ],
                  # Pyramid
                  5: [ [0, 1], [1, 2], [2, 3], [3, 0],
                       [0, 4], [1, 5], [2, 4], [3, 4] ],
                  # Wedge / Prism
                  6: [ [0, 1], [1, 2], [2, 0], [0, 3],
                       [2, 3], [3, 4], [4, 5], [5, 4] ],
                  # Hexahedron
                  8: [ [0, 1], [1, 2], [2, 3], [3, 0],
                       [0, 4], [1, 5], [2, 6], [3, 7],
                       [4, 5], [5, 6], [6, 7], [7, 4] ],
               }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in edge_map:
        raise ValueError(f'Error in edge_to_corner: elemType {elemType} is not supported')

    edges = edge_map[elemType % 100]

    try:
        return np.array(edges[edge], dtype=dtype)
    except KeyError:
        raise KeyError(f'Error in edge_to_corner: edge {edge} is not supported')


@cache
def face_to_edge(face: str, elemType: Union[str, int], dtype=int) -> np.ndarray:
    """ GMSH: Create faces from edges in the given direction
    """
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: {  'z-': np.array([  0,  1,   2,   3], dtype=dtype),
                         'y-': np.array([  0,  9,  -4,  -8], dtype=dtype),
                         'x+': np.array([  1, 10,  -5,  -9], dtype=dtype),
                         'y+': np.array([ -2, 10,   6, -11], dtype=dtype),
                         'x-': np.array([  8, -7, -11,   3], dtype=dtype),
                         'z+': np.array([  4,  5,   6,   7], dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_edge: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][face]
    except KeyError:
        raise KeyError(f'Error in face_to_edge: face {face} is not supported')


@cache
def face_to_corner(face, elemType: Union[str, int], dtype=int) -> np.ndarray:
    """ GMSH: Get points on faces in the given direction
    """
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: {  'z-': np.array([  0,  1,   2,   3], dtype=dtype),
                         'y-': np.array([  0,  1,   5,   4], dtype=dtype),
                         'x+': np.array([  1,  2,   6,   5], dtype=dtype),
                         'y+': np.array([  2,  6,   7,   3], dtype=dtype),
                         'x-': np.array([  0,  4,   7,   3], dtype=dtype),
                         'z+': np.array([  4,  5,   6,   7], dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_corner: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][face]
    except KeyError:
        raise KeyError(f'Error in face_to_corner: face {face} is not supported')


@cache
def face_to_cgns(face: str, elemType: Union[str, int], dtype=int) -> np.ndarray:
    """ CGNS: Get points on faces in the given direction
    """
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: {'z-': np.array([  0,  3,  2,  1], dtype=dtype),
                       'y-': np.array([  0,  1,  5,  4], dtype=dtype),
                       'x+': np.array([  1,  2,  6,  5], dtype=dtype),
                       'y+': np.array([  2,  3,  7,  6], dtype=dtype),
                       'x-': np.array([  0,  4,  7,  3], dtype=dtype),
                       'z+': np.array([  4,  5,  6,  7], dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_cgns: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][face]
    except KeyError:
        raise KeyError(f'Error in face_to_cgns: face {face} is not supported')


@cache
def flip_s2m(N: int, flip: int) -> np.ndarray:
    # Create grid index arrays for the rows and columns
    p = np.arange(N)
    q = np.arange(N)

    # Create a meshgrid of row (p) and column (q) indices
    p_grid, q_grid = np.meshgrid(p, q)

    # Map row and column indices based on flip logic
    # WARNING: FOR SOME REASON, ONLY FLIP 1,3,4 IS USED WITH FACE_TO_NODES
    if flip == 0:
        return np.stack((q_grid        , p_grid        ), axis=-1)
    elif flip == 1:
        return np.stack((p_grid        , q_grid        ), axis=-1)
    elif flip == 2:
        return np.stack((N - p_grid - 1, q_grid        ), axis=-1)
    elif flip == 3:
        return np.stack((N - p_grid - 1, N - q_grid - 1), axis=-1)
    elif flip == 4:
        return np.stack((N - q_grid - 1, p_grid        ), axis=-1)
    else:
        raise ValueError('Flip must be an integer between 0 and 4')


@cache
def type_to_mortar_flip(elemType: Union[int, str]) -> dict[int, dict[int, int]]:
    """ Returns the flip map for a given element type
    """

    flipID_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: { 0: {1: 1, 2: 2, 3: 3, 4: 4},
                        1: {1: 2, 4: 1, 3: 4, 2: 3},
                        2: {3: 1, 4: 2, 1: 3, 2: 4},
                        3: {2: 1, 3: 2, 4: 3, 1: 4}}
                }

    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    if elemType % 100 not in flipID_map:
        raise ValueError(f'Error in type_to_mortar_flip: elemType {elemType} is not supported')

    try:
        return flipID_map[elemType % 100]
    except KeyError:
        raise KeyError(f'Error in type_to_mortar_flip: elemType {elemType} is not supported')


@cache
def face_to_nodes(face: str, elemType: int, nGeo: int) -> np.ndarray:
    """ Returns the tensor-product nodes associated with a face
    """
    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    order     = nGeo
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: { 'z-':              LINMAP(elemType, order=order)[:    , :    , 0    ],
                        'y-': np.transpose(LINMAP(elemType, order=order)[:    , 0    , :    ]),
                        'x+': np.transpose(LINMAP(elemType, order=order)[order, :    , :    ]),
                        'y+':              LINMAP(elemType, order=order)[:    , order, :    ],
                        'x-':              LINMAP(elemType, order=order)[0    , :    , :    ],
                        'z+': np.transpose(LINMAP(elemType, order=order)[:    , :    , order])}

                }
    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_cgns: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][face]
    except KeyError:
        raise KeyError(f'Error in face_to_cgns: face {face} is not supported')


# > Not cacheable, we pass elemNode[np.ndarray]
def dir_to_nodes(dir: str, elemType: Union[str, int], elemNodes: np.ndarray, nGeo: int) -> np.ndarray:
    """ Returns the tensor-product nodes associated with a face
    """
    if isinstance(elemType, str):
        elemType = mesh_vars.ELEMTYPE.name[elemType]

    order     = nGeo
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: { 'z-':              elemNodes[:    , :    , 0    ],
                        'y-': np.transpose(elemNodes[:    , 0    , :    ]),
                        'x+': np.transpose(elemNodes[order, :    , :    ]),
                        'y+':              elemNodes[:    , order, :    ],
                        'x-':              elemNodes[0    , :    , :    ],
                        'z+': np.transpose(elemNodes[:    , :    , order])}
                 }
    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_cgns: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][dir]
    except KeyError:
        raise KeyError(f'Error in face_to_cgns: face {dir} is not supported')


# > Not cacheable, we pass mesh[meshio.Mesh]
def count_elems(mesh: meshio.Mesh) -> int:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    nElems = 0
    for _, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)
        nElems += ioelems.shape[0]
    return nElems


# > Not cacheable, we pass mesh[meshio.Mesh]
def calc_elem_bary(mesh: meshio.Mesh) -> np.ndarray:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    # Only consider three-dimensional types
    elem_cells = []
    for elemType in mesh.cells_dict:
        if any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            elem_cells.append(mesh.get_cells_type(elemType))

    # Flatten the list of cells (concatenate all cells into one array)
    all_cells = np.concatenate(elem_cells, axis=0)

    # Calculate the centroid (mean of coordinates) for all cells at once
    return np.mean(mesh_vars.mesh.points[all_cells], axis=1)


@cache
def LINTEN(elemType: int, order: int = 1) -> np.ndarray:
    """ CGNS -> IJK ordering for element corner nodes
    """
    # Local imports ----------------------------------------
    # from pyhope.io.io_cgns import genHEXMAPCGNS
    # from pyhope.io.io_vtk import genHEXMAPVTK
    from pyhope.io.io_meshio import HEXMAPMESHIO
    # ------------------------------------------------------
    match elemType:
        # Straight-sided elements, hard-coded
        case 104:  # Tetraeder
            return np.array([0, 1, 2, 3])
        case 105:  # Pyramid
            return np.array([0, 1, 3, 2, 4])
        case 106:  # Prism
            return np.array([0, 1, 2, 3, 4, 5])
        case 108:  # Hexaeder
            return np.array([0, 1, 3, 2, 4, 5, 7, 6])
        # Curved elements, use mapping
        case 208:  # Hexaeder
            # > HEXTEN : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
            # > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)

            # # CGNS
            # _, HEXTEN = HEXMAPCGNS(order+1)

            # # VTK
            # _, HEXTEN = HEXMAPVTK(order+1)

            # MESHIO
            _, HEXTEN = HEXMAPMESHIO(order+1)
            return HEXTEN
        case _:  # Default
            print('Error in LINMAP, unknown elemType')
            sys.exit(1)


@cache
def LINMAP(elemType: int, order: int = 1) -> npt.NDArray[np.int32]:
    """ CGNS -> IJK ordering for element corner nodes
    """
    # Local imports ----------------------------------------
    # from pyhope.io.io_cgns import HEXMAPCGNS
    # from pyhope.io.io_vtk import HEXMAPVTK
    from pyhope.io.io_meshio import HEXMAPMESHIO
    # ------------------------------------------------------
    match elemType:
        # Straight-sided elements, hard-coded
        case 104:  # Tetraeder
            sys.exit(1)
            return np.array([0, 1, 2, 3])
        case 105:  # Pyramid
            sys.exit(1)
            return np.array([0, 1, 3, 2, 4])
        case 106:  # Prism
            sys.exit(1)
            return np.array([0, 1, 2, 3, 4, 5])
        case 108:  # Hexaeder
            linmap = np.zeros((2, 2, 2), dtype=int)
            indices = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1) ]
            for i, index in enumerate(indices):
                linmap[index] = i
            return linmap

        # Curved elements, use mapping
        case 208:  # Hexaeder
            # > HEXTEN : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
            # > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)

            # # CGNS
            # HEXMAP, _ = HEXMAPCGNS(order+1)

            # # VTK
            # HEXMAP, _ = HEXMAPVTK(order+1)

            # MESHIO
            HEXMAP, _ = HEXMAPMESHIO(order+1)
            return HEXMAP
        case _:  # Default
            print('Error in LINMAP, unknown elemType')
            sys.exit(1)
