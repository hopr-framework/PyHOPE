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
from typing import Union
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def faces(elemType: Union[int, str]) -> list:
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


def face_to_cgns(face, elemType: Union[str, int], dtype=int) -> np.ndarray:
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


def count_elems(mesh: meshio._mesh.Mesh) -> int:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    nElems = 0
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)
        nElems += ioelems.shape[0]
    return nElems


def centeroidnp(coords: np.ndarray) -> np.ndarray:
    """ Compute the centroid (barycenter) of a set of coordinates
    """
    length = coords.shape[0]
    sum_x  = np.sum(coords[:, 0])
    sum_y  = np.sum(coords[:, 1])
    sum_z  = np.sum(coords[:, 2])
    return np.array([sum_x/length, sum_y/length, sum_z/length])


def calc_elem_bary(mesh: meshio._mesh.Mesh) -> list:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    nElems   = count_elems(mesh)
    elemBary = [np.ndarray(3)] * nElems
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)

        for elemID, cell in enumerate(ioelems):
            elemBary[elemID] = centeroidnp(mesh_vars.mesh.points[cell])
    return elemBary
