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
from typing import Tuple
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


def facePointMatrixFill(matrix: np.ndarray, start: int, end: int, count: int, orient: bool) -> tuple[np.ndarray, int]:
    """ Fill the 2D matrix representing the inner points of each faces
    """
    if end <= start:
        return matrix, count

    # Fill the corner nodes first
    matrix[start, start] = count
    count += 1
    if orient:
        matrix[start, end  ] = count
        count += 1
        matrix[end  , end  ] = count
        count += 1
        matrix[end  , start] = count
        count += 1
    else:
        matrix[end  , start] = count
        count += 1
        matrix[end  , end  ] = count
        count += 1
        matrix[start, end  ] = count
        count += 1

    # Now, fill the edges
    if orient:
        for i in range(start+1, end):
            matrix[start, i    ] = count
            count += 1
        for i in range(start+1, end):
            matrix[i    , end  ] = count
            count += 1
        for i in range(end-1, start, -1):
            matrix[end  , i    ] = count
            count += 1
        for i in range(end-1, start, -1):
            matrix[i    , start] = count
            count += 1
    else:
        for i in range(start+1, end):
            matrix[i    , start] = count
            count += 1
        for i in range(start+1, end):
            matrix[end  , i    ] = count
            count += 1
        for i in range(end-1, start, -1):
            matrix[i    , end  ] = count
            count += 1
        for i in range(end-1, start, -1):
            matrix[start, i    ] = count
            count += 1

    return matrix, count


def facePointMatrix(order: int, pos: int, orient: bool = True) -> np.ndarray:
    """ Return the 2D index of the inner points of each faces
    """
    # Create a matrix of the required size
    matrix = np.zeros((order, order), dtype=int)

    count  = 0
    # Fill the matrix recursively
    for i in range(np.floor(order/2).astype(int)):
        matrix, count = facePointMatrixFill(matrix, i, order-i-1, count, orient)

    # Fill the middle point if uneven order
    if order % 2 != 0:
        matrix[np.floor(order/2).astype(int), np.floor(order/2).astype(int)] = count
    return np.argwhere(matrix == pos)[0]


def edgePointMESHIO(start: int, end: int, edge: int, node: int) -> np.ndarray:
    """ Traverse over all 12 edges of the hexahedron
    """
    match edge:
        case 0:
            return np.array([node          , start , start  ], dtype=int)
        case 1:
            return np.array([start         , node  , start  ], dtype=int)
        case 2:
            return np.array([start         , start , node   ], dtype=int)
        case 3:
            return np.array([end           , node  , start  ], dtype=int)
        case 4:
            return np.array([end           , start , node   ], dtype=int)
        case 5:
            return np.array([end+start-node, end   , start  ], dtype=int)
        case 6:
            return np.array([end           , end   , node   ], dtype=int)
        case 7:
            return np.array([start         , end   , node   ], dtype=int)
        case 8:
            return np.array([node          , start , end    ], dtype=int)
        case 9:
            return np.array([start         , node  , end    ], dtype=int)
        case 10:
            return np.array([end           , node  , end    ], dtype=int)
        case 11:
            return np.array([end+start-node, end   , end    ], dtype=int)
        case _:
            sys.exit(1)


def facePointMESHIO(start: int, end: int, face: int, pos: int) -> np.ndarray:
    """ Translate the 1D position of each of the 6 hexahedron faces to each 2D index
    """
    match face:
        case 0:
            index = facePointMatrix(end-start-1, pos, orient=True)
            return np.array([start+index[0]+1 , start+index[1]+1 , start           ], dtype=int)
        case 1:
            index = facePointMatrix(end-start-1, pos, orient=False)
            return np.array([start+index[0]+1 , start            , start+index[1]+1], dtype=int)
        case 2:
            index = facePointMatrix(end-start-1, pos, orient=True)
            return np.array([start            , start+index[0]+1 , start+index[1]+1], dtype=int)
        case 3:
            index = facePointMatrix(end-start-1, pos, orient=False)
            return np.array([end              , start+index[0]+1 , start+index[1]+1], dtype=int)
        case 4:
            index = facePointMatrix(end-start-1, pos, orient=False)
            return np.array([end  -index[0]-1 , end              , start+index[1]+1], dtype=int)
        case 5:
            index = facePointMatrix(end-start-1, pos, orient=False)
            return np.array([start+index[0]+1 , start+index[1]+1 , end             ], dtype=int)
        case _:
            sys.exit(1)


@cache
def HEXMAPMESHIO(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """ MESHIO -> IJK ordering for high-order hexahedrons
        > HEXTEN : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
        > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)
    """
    map = np.zeros((order, order, order), dtype=int)

    if order == 1:
        map[0, 0, 0] = 0
        tensor       = map
        return map, tensor

    if order == 2:
        # Python indexing, 1 -> 0
        map[0 , 0 , 0 ] = 1
        map[1 , 0 , 0 ] = 2
        map[1 , 1 , 0 ] = 3
        map[0 , 1 , 0 ] = 4
        map[0 , 0 , 1 ] = 5
        map[1 , 0 , 1 ] = 6
        map[1 , 1 , 1 ] = 7
        map[0 , 1 , 1 ] = 8
        map -= 1

        # Reshape into 1D array, tensor-product style
        tensor = []
        for k in range(order):
            for j in range(order):
                for i in range(order):
                    tensor.append(int(map[i, j, k]))

        return map, np.asarray(tensor)

    count = 0

    # Fill the cube recursively from the outside to the inside
    for iOrder in range(np.floor(order/2).astype(int)):
        # Principal vertices
        map[iOrder         , iOrder         , iOrder        ] = count+1
        map[order-iOrder-1 , iOrder         , iOrder        ] = count+2
        map[order-iOrder-1 , order-iOrder-1 , iOrder        ] = count+3
        map[iOrder         , order-iOrder-1 , iOrder        ] = count+4
        map[iOrder         , iOrder         , order-iOrder-1] = count+5
        map[order-iOrder-1 , iOrder         , order-iOrder-1] = count+6
        map[order-iOrder-1 , order-iOrder-1 , order-iOrder-1] = count+7
        map[iOrder         , order-iOrder-1 , order-iOrder-1] = count+8
        count += 8

        # Loop over all edges
        for iEdge in range(12):
            for iNode in range(iOrder+1, order-iOrder-1):
                # Assemble mapping to tuple
                count += 1
                edge  = edgePointMESHIO(iOrder, order-iOrder-1, iEdge, iNode)
                index = (int(edge[0]), int(edge[1]), int(edge[2]))
                map[index] = count

        # Internal points of upstanding faces
        for iFace in range(6):
            for pos in range((order-2*iOrder-2)**2):
                # Assemble mapping to tuple, top  quadrangle -> z = order
                count += 1
                index = facePointMESHIO(iOrder, order-iOrder-1, iFace, pos)
                index = (int(index[0]), int(index[1]), int(index[2]))
                map[index] = count

    # Fill the middle point if uneven order
    if order % 2 != 0:
        index = (np.floor(order/2).astype(int), np.floor(order/2).astype(int), np.floor(order/2).astype(int))
        map[index] = count+1

    # Python indexing, 1 -> 0
    map -= 1

    # Reshape into 1D array, tensor-product style
    tensor = []
    for k in range(order):
        for j in range(order):
            for i in range(order):
                tensor.append(int(map[i, j, k]))

    return map, np.asarray(tensor)
