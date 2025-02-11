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


@cache
def ElemTypes(num: int) -> dict[str, str | int]:
    types = [{ 'ElemTypeCGNS': 'ElementTypeNull', 'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 0
             { 'ElemTypeCGNS': 'ElementTypeUser', 'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 1
             { 'ElemTypeCGNS': 'NODE',            'ElemTypeMeshIO': 'vertex',       'Nodes': 1  },  # 2
             { 'ElemTypeCGNS': 'BAR_2',           'ElemTypeMeshIO': 'line',         'Nodes': 2  },  # 3
             { 'ElemTypeCGNS': 'BAR_3',           'ElemTypeMeshIO': 'line3',        'Nodes': 3  },  # 4
             { 'ElemTypeCGNS': 'TRI_3',           'ElemTypeMeshIO': 'triangle',     'Nodes': 3  },  # 5
             { 'ElemTypeCGNS': 'TRI_6',           'ElemTypeMeshIO': 'triangle6',    'Nodes': 6  },  # 6
             { 'ElemTypeCGNS': 'QUAD_4',          'ElemTypeMeshIO': 'quad',         'Nodes': 4  },  # 7
             { 'ElemTypeCGNS': 'QUAD_8',          'ElemTypeMeshIO': 'quad8',        'Nodes': 8  },  # 8
             { 'ElemTypeCGNS': 'QUAD_9',          'ElemTypeMeshIO': 'quad9',        'Nodes': 9  },  # 9
             { 'ElemTypeCGNS': 'TETRA_4',         'ElemTypeMeshIO': 'tetra',        'Nodes': 4  },  # 10
             { 'ElemTypeCGNS': 'TETRA_10',        'ElemTypeMeshIO': 'tetra10',      'Nodes': 10 },  # 11
             { 'ElemTypeCGNS': 'PYRA_5',          'ElemTypeMeshIO': 'pyramid',      'Nodes': 5  },  # 12
             { 'ElemTypeCGNS': 'PYRA_14',         'ElemTypeMeshIO': 'pyramid14',    'Nodes': 14 },  # 13
             { 'ElemTypeCGNS': 'PENTA_6',         'ElemTypeMeshIO': 'wedge',        'Nodes': 6  },  # 14
             { 'ElemTypeCGNS': 'PENTA_15',        'ElemTypeMeshIO': 'wedge15',      'Nodes': 15 },  # 15
             { 'ElemTypeCGNS': 'PENTA_18',        'ElemTypeMeshIO': 'wedge18',      'Nodes': 18 },  # 16
             { 'ElemTypeCGNS': 'HEXA_8',          'ElemTypeMeshIO': 'hexahedron',   'Nodes': 8  },  # 17
             { 'ElemTypeCGNS': 'HEXA_20',         'ElemTypeMeshIO': 'hexahedron20', 'Nodes': 20 },  # 18
             { 'ElemTypeCGNS': 'HEXA_27',         'ElemTypeMeshIO': 'hexahedron27', 'Nodes': 27 },  # 19
             { 'ElemTypeCGNS': 'MIXED',           'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 20
             { 'ElemTypeCGNS': 'PYRA_13',         'ElemTypeMeshIO': 'pyramid13',    'Nodes': 13 },  # 21
             { 'ElemTypeCGNS': 'NGON_n',          'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 22
             { 'ElemTypeCGNS': 'NFACE_n',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 23
             { 'ElemTypeCGNS': 'BAR_4',           'ElemTypeMeshIO': 'line4',        'Nodes': 4  },  # 24
             { 'ElemTypeCGNS': 'TRI_9',           'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 25
             { 'ElemTypeCGNS': 'TRI_10',          'ElemTypeMeshIO': 'triangle10',   'Nodes': 10 },  # 26
             { 'ElemTypeCGNS': 'QUAD_12',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 27
             { 'ElemTypeCGNS': 'QUAD_16',         'ElemTypeMeshIO': 'quad16',       'Nodes': 16 },  # 28
             { 'ElemTypeCGNS': 'TETRA_16',        'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 29
             { 'ElemTypeCGNS': 'TETRA_20',        'ElemTypeMeshIO': 'tetra20',      'Nodes': 20 },  # 30
             { 'ElemTypeCGNS': 'PYRA_21',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 31
             { 'ElemTypeCGNS': 'PYRA_29',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 32
             { 'ElemTypeCGNS': 'PYRA_30',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 33
             { 'ElemTypeCGNS': 'PENTA_24',        'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 34
             { 'ElemTypeCGNS': 'PENTA_38',        'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 35
             { 'ElemTypeCGNS': 'PENTA_40',        'ElemTypeMeshIO': 'wedge40',      'Nodes': 40 },  # 36
             { 'ElemTypeCGNS': 'HEXA_32',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 37
             { 'ElemTypeCGNS': 'HEXA_56',         'ElemTypeMeshIO': 'Null',         'Nodes': 0  },  # 38
             { 'ElemTypeCGNS': 'HEXA_64',         'ElemTypeMeshIO': 'hexahedron64', 'Nodes': 64 },  # 39
             { 'ElemTypeCGNS': 'Null',            'ElemTypeMeshIO': 'hexahedron24', 'Nodes': 24 },  # 40
             ]
    return types[num]


# def spiral_matrix(n: int) -> np.ndarray:
#     """ Print a spiral matrix of order n
#         > https://rosettacode.org/wiki/Spiral_matrix#Simple_solution
#     """
#     m = [[0] * n for _ in range(n)]
#     dx, dy  = [0, 1, 0, -1], [1, 0, -1, 0]
#     x, y, c =  0, -1, 1
#     for i in range(n + n - 1):
#         for j in range((n + n - i) // 2):
#             x += dx[i % 4]
#             y += dy[i % 4]
#             m[x][y] = c
#             c += 1
#     return np.array(m)
#
#
# def edgePointCGNS(order: int, edge: int, node: int) -> np.ndarray:
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
# def HEXMAPCGNS(order: int) -> Tuple[np.ndarray, np.ndarray]:
#     """ CGNS -> IJK ordering for high-order hexahedrons
#         > Loosely based on [Gmsh] "generatePointsHexCGNS"
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
#             edge  = edgePointCGNS(order-1, iFace, iNode)
#             index = (int(edge[0]), int(edge[1]), 0)
#             map[index] = count
#
#     # Internal points of mounting edges
#     for iFace in range(4):
#         edge  = edgePointCGNS(order-1, (iFace+3) % 4, order-1)
#         for iNode in range(1, order-1):
#             # Assemble mapping to tuple, mounting edges -> z ascending
#             count += 1
#             index = (int(edge[0]), int(edge[1]), iNode)
#             map[index] = count
#
#     # Internal points of top quadrangle edges
#     for iFace in range(4):
#         for iNode in range(1, order-1):
#             # Assemble mapping to tuple, top  quadrangle -> z = order
#             count += 1
#             edge  = edgePointCGNS(order-1, iFace, iNode)
#             index = (int(edge[0]), int(edge[1]), order-1)
#             map[index] = count
#
#     # Internal points of base quadrangle
#     k    = 0
#     # Fill in spirals
#     face = spiral_matrix(order-2)
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             index = (i+1  , j+1  , k    )
#             map[index] = count + face[j, i]
#     count += (order-2)**2
#
#     # Internal points of upstanding faces
#     # > y- face
#     k    = 0
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             index = (i+1  , k    , j+1  )
#             map[index] = count + face[j, i]
#     count += (order-2)**2
#
#     # > x+ face
#     k    = order-1
#     for j in range(order-2):
#         for i in range(order-2):
#             index = (k    , i+1  , j+1  )
#             map[index] = count + face[j, i]
#     count += (order-2)**2
#
#     # > y+ face
#     k      = order-1
#     face_r = np.rot90(face, k=2, axes=(1, 0))
#     for j in range(order-2):
#         for i in range(order-2):
#             index = (i+1  , k    , j+1  )
#             map[index] = count + face_r[order-3-j, i]
#     count += (order-2)**2
#
#     # > x- face
#     k      = 0
#     for j in range(order-2):
#         for i in range(order-2):
#             index = (k    , i+1  , j+1  )
#             map[index] = count + face_r[order-3-j, i]
#     count += (order-2)**2
#
#     # Internal points of top  quadrangle
#     k    = order-1
#     # Fill in spirals
#     face = spiral_matrix(order-2)
#     # Fill the map
#     for j in range(order-2):
#         for i in range(order-2):
#             index = (i+1  , j+1  , k    )
#             map[index] = count + face[j, i]
#     count += (order-2)**2
#
#     # Internal volume points as a succession of internal planes
#     for k in range(1, order-1):
#         for j in range(order-2):
#             for i in range(order-2):
#                 index = (i+1  , j+1  , k    )
#                 map[index] = count + face[j, i]
#         count += (order-2)**2
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
