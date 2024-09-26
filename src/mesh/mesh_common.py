#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of UVWXYZ
#
# Copyright (c) 2022-2024 Andrea Beck
#
# UVWXYZ is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# UVWXYZ is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# UVWXYZ. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import sys
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


def faces() -> list:
    """ Return a list of all sides of a hexahedron
    """
    return ['z-', 'y-', 'x+', 'y+', 'x-', 'z+']


def edge_to_dir(edge: int) -> int:
    """ GMSH: Create edges from points in the given direction
    """
    match edge:
        case 0 | 2 |  4 |  6:
            return 0
        case 1 | 3 |  5 |  7:
            return 1
        case 8 | 9 | 10 | 11:
            return 2
        case _:
            print('Error in edge_to_dir, unknown edge')
            sys.exit()


def face_to_edge(face, dtype=int) -> np.ndarray:
    """ GMSH: Create faces from edges in the given direction
    """
    match face:
        case 'z-':
            return np.array([  0,  1,   2,   3], dtype=dtype)
        case 'y-':
            return np.array([  0,  9,  -4,  -8], dtype=dtype)
        case 'x+':
            return np.array([  1, 10,  -5,  -9], dtype=dtype)
        case 'y+':
            return np.array([ -2, 10,   6, -11], dtype=dtype)
        case 'x-':
            return np.array([  8, -7, -11,   3], dtype=dtype)
        case 'z+':
            return np.array([  4,  5,   6,   7], dtype=dtype)
        case _:  # Default
            print('Error in face_to_edge, unknown face')
            sys.exit()


def face_to_corner(face, dtype=int) -> np.ndarray:
    """ GMSH: Get points on faces in the given direction
    """
    match face:
        case 'z-':
            return np.array([  0,  1,   2,   3], dtype=dtype)
        case 'y-':
            return np.array([  0,  1,   5,   4], dtype=dtype)
        case 'x+':
            return np.array([  1,  2,   6,   5], dtype=dtype)
        case 'y+':
            return np.array([  2,  6,   7,   3], dtype=dtype)
        case 'x-':
            return np.array([  0,  4,   7,   3], dtype=dtype)
        case 'z+':
            return np.array([  4,  5,   6,   7], dtype=dtype)
        case _:  # Default
            print('Error in face_to_corner, unknown face')
            sys.exit()


def face_to_cgns(face, dtype=int) -> np.ndarray:
    """ CGNS: Get points on faces in the given direction
    """
    match face:
        case 'z-':
            return np.array([  0,  3,   2,   1], dtype=dtype)
        case 'y-':
            return np.array([  0,  1,   5,   4], dtype=dtype)
        case 'x+':
            return np.array([  1,  2,   6,   5], dtype=dtype)
        case 'y+':
            return np.array([  2,  3,   7,   6], dtype=dtype)
        case 'x-':
            return np.array([  0,  4,   7,   3], dtype=dtype)
        case 'z+':
            return np.array([  4,  5,   6,   7], dtype=dtype)
        case _:  # Default
            print('Error in face_to_corner, unknown face')
            sys.exit()


def count_elems(mesh: meshio._mesh.Mesh) -> int:
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    nElems = 0
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEM.type.keys()):
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
    import src.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    nElems   = count_elems(mesh)
    elemBary = [np.ndarray(3)] * nElems
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEM.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)

        for elemID, cell in enumerate(ioelems):
            elemBary[elemID] = centeroidnp(mesh_vars.mesh.points[cell])
    return elemBary
