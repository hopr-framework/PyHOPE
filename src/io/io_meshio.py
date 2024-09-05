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
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def edgePointMESHIO(order: int, edge: int, node: int) -> np.ndarray:
    # TODO: SOMEONE EXPLAIN THIS LOGIC TO ME
    match edge:
        case 0:
            return np.array([node , 0    , 0    ], dtype=int)
        case 1:
            return np.array([0    , node , 0    ], dtype=int)
        case 2:
            return np.array([0    , 0    , node ], dtype=int)
        case 3:
            return np.array([order, node , 0    ], dtype=int)
        case 4:
            return np.array([order, 0    , node ], dtype=int)
        case 5:
            return np.array([node , order, 0    ], dtype=int)
        case 6:
            return np.array([order, order, node ], dtype=int)
        case 7:
            return np.array([0    , order, node ], dtype=int)
        case 8:
            return np.array([node , 0    , order], dtype=int)
        case 9:
            return np.array([0    , node , order], dtype=int)
        case 10:
            return np.array([order, node , order], dtype=int)
        case 11:
            return np.array([node , order, order], dtype=int)
        case _:
            sys.exit()


def facePointMESHIO(order: int, face: int, iNode: int, jNode: int) -> np.ndarray:
    # TODO: SOMEONE EXPLAIN THIS LOGIC TO ME
    match face:
        case 0:
            return np.array([iNode , jNode , 0     ], dtype=int)
        case 1:
            return np.array([iNode , 0     , jNode ], dtype=int)
        case 2:
            return np.array([0     , iNode , jNode ], dtype=int)
        case 3:
            return np.array([order , iNode , jNode ], dtype=int)
        case 4:
            return np.array([iNode , order , jNode ], dtype=int)
        case 5:
            return np.array([iNode , jNode , order ], dtype=int)
        case _:
            sys.exit()


def genHEXMAPMESHIO(order: int) -> None:
    """ MESHIO -> IJK ordering for high-order hexahedrons
        > Losely based on [Gmsh] "generatePointsHexCGNS"
        > [Jens Ulrich Kreber] "paraview-scripts/node_ordering.py"
    """
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    map = np.zeros((order, order, order), dtype=int)

    if order == 1:
        map[0, 0, 0] = 0
        mesh_vars.HEXMAP = map
        return None

    # Principal vertices
    map[0      , 0      , 0      ] = 1
    map[order-1, 0      , 0      ] = 2
    map[order-1, order-1, 0      ] = 3
    map[0      , order-1, 0      ] = 4
    map[0      , 0      , order-1] = 5
    map[order-1, 0      , order-1] = 6
    map[order-1, order-1, order-1] = 7
    map[0      , order-1, order-1] = 8

    if order == 2:
        # Python indexing, 1 -> 0
        map -= 1
        # Reshape into 1D array, tensor-product style
        tensor = []
        for k in range(order):
            for j in range(order):
                for i in range(order):
                    tensor.append(int(map[i, j, k]))

        mesh_vars.HEXMAP = tensor
        return None

    count = 8
    # Loop over all edges
    for iEdge in range(12):
        for iNode in range(1, order-1):
            # Assemble mapping to tuple
            count += 1
            edge  = edgePointMESHIO(order-1, iEdge, iNode)
            index = (int(edge[0]), int(edge[1]), int(edge[2]))
            map[index] = count

    # Internal points of upstanding faces
    for iFace in range(6):
        for j in range(1, order-1):
            for i in range(1, order-1):
                # Assemble mapping to tuple, top  quadrangle -> z = order
                count += 1
                edge  = facePointMESHIO(order-1, iFace, i, j)
                index = (int(edge[0]), int(edge[1]), int(edge[2]))
                map[index] = count

    # Internal volume points as a tensor product
    for k in range(1, order-1):
        for j in range(1, order-1):
            for i in range(1, order-1):
                count += 1
                index = (i  , j  , k  )
                map[index] = count

    # Python indexing, 1 -> 0
    map -= 1

    # Reshape into 1D array, tensor-product style
    tensor = []
    for k in range(order):
        for j in range(order):
            for i in range(order):
                tensor.append(int(map[i, j, k]))
    mesh_vars.HEXMAP = tensor

