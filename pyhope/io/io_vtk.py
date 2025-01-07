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
# from functools import cache
# from typing import Tuple
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


# def edgePointVTK(order: int, edge: int, node: int) -> np.ndarray:
#     match edge:
#         case 0:  # z- / base
#             return np.array([node      , 0         ], dtype=int)
#         case 1:  # y+ / base
#             return np.array([order     , node      ], dtype=int)
#         case 2:  # z+ / base
#             return np.array([order-node, order     ], dtype=int)
#         case 3:  # y- / base
#             return np.array([0         , order-node], dtype=int)
#         case _:
#             sys.exit(1)
#
#
# @cache
# def HEXMAPVTK(order: int) ->  -> Tuple[np.ndarray, np.ndarray]:
#     """ VTK -> IJK ordering for high-order hexahedrons
#         > Loosely based on [Gmsh] "generatePointsHexCGNS"
#         > [Jens Ulrich Kreber] "paraview-scripts/node_ordering.py"
#
#         > HEXTEN : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
#         > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)
#     """
#     map = np.zeros((order, order, order), dtype=int)
#
#     if order == 1:
#         map[0, 0, 0] = 0
#         tensor       = map
#         return map, tensor
#
#     # Principal vertices
#     map[0      , 0      , 0      ] = 1
#     map[order-1, 0      , 0      ] = 2
#     map[order-1, order-1, 0      ] = 3
#     map[0      , order-1, 0      ] = 4
#     map[0      , 0      , order-1] = 5
#     map[order-1, 0      , order-1] = 6
#     map[order-1, order-1, order-1] = 7
#     map[0      , order-1, order-1] = 8
#
#     if order == 2:
#         # Python indexing, 1 -> 0
#         map -= 1
#         # Reshape into 1D array, tensor-product style
#         tensor = []
#         for k in range(order):
#             for j in range(order):
#                 for i in range(order):
#                     tensor.append(int(map[i, j, k]))
#
#         return map, np.asarray(tensor)
#
#     # Internal points of base quadrangle edges (x-)
#     count = 8
#     for iFace in range(4):
#         for iNode in range(1, order-1):
#             # Assemble mapping to tuple, base quadrangle -> z = 0
#             count += 1
#             edge  = edgePointVTK(order-1, iFace, iNode)
#             index = (int(edge[0]), int(edge[1]), 0)
#             map[index] = count
#
#     # Internal points of top quadrangle edges
#     for iFace in range(4):
#         for iNode in range(1, order-1):
#             # Assemble mapping to tuple, top  quadrangle -> z = order
#             count += 1
#             edge  = edgePointVTK(order-1, iFace, iNode)
#             index = (int(edge[0]), int(edge[1]), order-1)
#             map[index] = count
#
#     # Internal points of mounting edges
#     for iFace in range(4):
#         edge  = edgePointVTK(order-1, (iFace+3) % 4, order-1)
#         for iNode in range(1, order-1):
#             # Assemble mapping to tuple, mounting edges -> z ascending
#             count += 1
#             index = (int(edge[0]), int(edge[1]), iNode)
#             map[index] = count
#     # > VTK9 swapped 3/4
#     # (0      ,0)
#     # for iNode in range(1, order-1):
#     #     # Assemble mapping to tuple, mounting edges -> z ascending
#     #     count += 1
#     #     index = (0           , 0           , iNode)
#     #     map[index] = count
#     # # (order-1,0)
#     # for iNode in range(1, order-1):
#     #     # Assemble mapping to tuple, mounting edges -> z ascending
#     #     count += 1
#     #     index = (order-1     , 0           , iNode)
#     #     map[index] = count
#     # # (order-1,order-1)
#     # for iNode in range(1, order-1):
#     #     # Assemble mapping to tuple, mounting edges -> z ascending
#     #     count += 1
#     #     index = (order-1     , order-1     , iNode)
#     #     map[index] = count
#     # # (order-1,order-1)
#     # for iNode in range(1, order-1):
#     #     # Assemble mapping to tuple, mounting edges -> z ascending
#     #     count += 1
#     #     index = (order-1     , order-1     , iNode)
#     #     map[index] = count
#
#     # WARNING:THIS IS HOW IT SHOULD BE
#     # Internal points of upstanding faces
#     # > x- face
#     k = 0
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             count += 1
#             index = (k    , i+1  , j+1  )
#             map[index] = count
#     # > x+ face
#     k = order-1
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             count += 1
#             index = (k    , i+1  , j+1  )
#             map[index] = count
#     # > y- face
#     k = 0
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             count += 1
#             index = (i+1  , k    , j+1  )
#             map[index] = count
#     # > y+ face
#     k = order-1
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             count += 1
#             index = (i+1  , k    , j+1  )
#             map[index] = count
#     # Internal points of base quadrangle (z-)
#     k = 0
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             count += 1
#             index = (i+1  , j+1  , k    )
#             map[index] = count
#
#     # Internal points of top  quadrangle (z+)
#     k = order-1
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             count += 1
#             index = (i+1  , j+1  , k    )
#             map[index] = count
#
#     # # FIXME: THIS IS HOW MESHIO GIVES IT
#     # # Internal points of base quadrangle (z-)
#     # k = 0
#     # # Fill the map
#     # for j in range(order-2):
#     #     for i in range(order-2):
#     #         count += 1
#     #         index = (i+1  , j+1  , k    )
#     #         map[index] = count
#     # # > y- face
#     # k = 0
#     # # Fill the map
#     # for j in range(order-2):
#     #     for i in range(order-2):
#     #         count += 1
#     #         index = (i+1  , k    , j+1  )
#     #         map[index] = count
#     # # > x+ face
#     # k = order-1
#     # # Fill the map
#     # for j in range(order-2):
#     #     for i in range(order-2):
#     #         count += 1
#     #         index = (k    , i+1  , j+1  )
#     #         map[index] = count
#     # # > y+ face
#     # k = order-1
#     # # Fill the map
#     # for j in range(order-2):
#     #     for i in range(order-2):
#     #         count += 1
#     #         index = (i+1  , k    , j+1  )
#     #         map[index] = count
#     # # > x- face
#     # k = 0
#     # # Fill the map
#     # for j in range(order-2):
#     #     for i in range(order-2):
#     #         count += 1
#     #         index = (k    , i+1  , j+1  )
#     #         map[index] = count
#     # # Internal points of top  quadrangle (z+)
#     # k = order-1
#     # # Fill the map
#     # for j in range(order-2):
#     #     for i in range(order-2):
#     #         count += 1
#     #         index = (i+1  , j+1  , k    )
#     #         map[index] = count
#
#     # Internal volume points as a tensor product
#     for k in range(1, order-1):
#         for j in range(1, order-1):
#             for i in range(1, order-1):
#                 count += 1
#                 index = (i  , j  , k  )
#                 map[index] = count
#
#     # Python indexing, 1 -> 0
#     map -= 1
#
#     # Reshape into 1D array, tensor-product style
#     tensor = []
#     for k in range(order):
#         for j in range(order):
#             for i in range(order):
#                 tensor.append(int(map[i, j, k]))
#
#     return map, np.asarray(tensor)
