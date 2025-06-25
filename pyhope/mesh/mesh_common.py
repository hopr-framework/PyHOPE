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
from typing import Union, Tuple, Any
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
# Instantiate ELEMTYPE
elemTypeClass = mesh_vars.ELEMTYPE()
# ==================================================================================================================================


@cache
def faces(elemType: Union[int, str]) -> list[str]:
    """ Return a list of all sides of an element
    """
    faces_map = {  # Tetrahedron
                   4: ['z-', 'y-', 'x+', 'x-'            ],
                   # Pyramid
                   5: ['z-', 'y-', 'x+', 'y+', 'x-'      ],
                   # Wedge / Prism
                   6: ['y-', 'x+', 'x-', 'z-', 'z+'      ],
                   # Hexahedron
                   8: ['z-', 'y-', 'x+', 'y+', 'x-', 'z+']
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in faces: elemType {elemType} is not supported')

    return faces_map[elemType % 100]


@cache
def edges(elemType: Union[int, str]) -> list[int]:
    """ Return a list of all edges of an element
    """
    edges_map = {  # Tetrahedron
                   4: [0, 1, 2, 3, 4, 5],
                   # Pyramid
                   5: [0, 1, 2, 3, 4, 5, 6, 7],
                   # Wedge / Prism
                   6: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                   # Hexahedron
                   8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in edges_map:
        raise ValueError(f'Error in edges: elemType {elemType} is not supported')

    return edges_map[elemType % 100]


@cache
def edge_to_dir(edge: int, elemType: Union[int, str]) -> int:
    """ GMSH: Create edges from points in the given direction
    """
    eps = np.finfo(np.float64).eps
    dir_map  = {  # Tetrahedron
                  # Pyramid
                  # Wedge / Prism
                  # Hexahedron
                  8: {  0:  eps,  2:  eps,  4: eps,  6:   eps,  # Direction 0
                        1:   1.,  3:   1.,  5:  1.,  7:    1.,  # Direction 1
                        8:   2.,  9:   2., 10:  2., 11:    2.}  # Direction 2
               }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in dir_map:
        raise ValueError(f'Error in edge_to_direction: elemType {elemType} is not supported')

    dir = dir_map[elemType % 100]

    try:
        return (np.rint(abs(dir[edge]))).astype(int)
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
        elemType = elemTypeClass.name[elemType]

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
                   8: {  'z-': np.array((  0,  1,   2,   3), dtype=dtype),
                         'y-': np.array((  0,  9,  -4,  -8), dtype=dtype),
                         'x+': np.array((  1, 10,  -5,  -9), dtype=dtype),
                         'y+': np.array(( -2, 10,   6, -11), dtype=dtype),
                         'x-': np.array((  8, -7, -11,   3), dtype=dtype),
                         'z+': np.array((  4,  5,   6,   7), dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

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
                   8: {  'z-': np.array((  0,  1,   2,   3), dtype=dtype),
                         'y-': np.array((  0,  1,   5,   4), dtype=dtype),
                         'x+': np.array((  1,  2,   6,   5), dtype=dtype),
                         'y+': np.array((  2,  6,   7,   3), dtype=dtype),
                         'x-': np.array((  0,  4,   7,   3), dtype=dtype),
                         'z+': np.array((  4,  5,   6,   7), dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

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
                   4: {'z-': np.array((  0,  2,  1    ), dtype=dtype),
                       'y-': np.array((  0,  1,  3    ), dtype=dtype),
                       'x+': np.array((  1,  2,  3    ), dtype=dtype),
                       'x-': np.array((  2,  0,  3    ), dtype=dtype)},
                   # Pyramid
                   5: {'z-': np.array((  0,  3,  2,  1), dtype=dtype),
                       'y-': np.array((  0,  1,  4    ), dtype=dtype),
                       'x+': np.array((  1,  2,  4    ), dtype=dtype),
                       'y+': np.array((  2,  3,  4    ), dtype=dtype),
                       'x-': np.array((  3,  0,  4    ), dtype=dtype)},
                   # Wedge / Prism
                   6: {'y-': np.array((  0,  1,  4,  3), dtype=dtype),
                       'x+': np.array((  1,  2,  5,  4), dtype=dtype),
                       'x-': np.array((  2,  0,  3,  5), dtype=dtype),
                       'z-': np.array((  0,  2,  1    ), dtype=dtype),
                       'z+': np.array((  3,  4,  5    ), dtype=dtype)},
                   # Hexahedron
                   8: {'z-': np.array((  0,  3,  2,  1), dtype=dtype),
                       'y-': np.array((  0,  1,  5,  4), dtype=dtype),
                       'x+': np.array((  1,  2,  6,  5), dtype=dtype),
                       'y+': np.array((  2,  3,  7,  6), dtype=dtype),
                       'x-': np.array((  0,  4,  7,  3), dtype=dtype),
                       'z+': np.array((  4,  5,  6,  7), dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_cgns: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][face]
    except KeyError:
        raise KeyError(f'Error in face_to_cgns: face {face} is not supported')


# @dataclass
# class FaceOrdering:
#     side_type: str
#     nGeo     : int
#     order    : np.ndarray = field(init=False)
#
#     def __post_init__(self):
#         self.order = self.compute_ordering()
#
#     def compute_ordering(self) -> np.ndarray:
@cache
def FaceOrdering(side_type: str, order: int) -> np.ndarray:
    """
    Compute the permutation ordering to convert from tensor-product ordering
    to meshio ordering for a face of a given type ('quad' or 'triangle')
    and polynomial order nGeo.

    For quadrilaterals, total nodes = (nGeo+1)**2.
      - For nGeo==1, the natural ordering is [0, 1, 2, 3].
      - For nGeo>1, the ordering is:
          * Corners: bottom-left, bottom-right, top-right, top-left;
          * Then the bottom edge (excluding corners, left-to-right);
          * Then the right  edge (excluding corners, bottom-to-top);
          * Then the top    edge (excluding corners, right-to-left);
          * Then the left   edge (excluding corners, top-to-bottom);
          * Finally, the interior nodes in row-major order.

    For triangles, total nodes = (nGeo+1)*(nGeo+2)//2.
      - For nGeo==1, the natural ordering is [0, 1, 2].
      - For nGeo>1, we generate the tensor ordering as all (i,j) pairs
        with i+j <= nGeo (in lexicographical order) and then reorder so that:
          * Vertices come first: (0,0), (nGeo,0), (0,nGeo);
          * Followed by edge nodes (in order along each edge);
          * And then the interior nodes in their natural order.
    """
    if side_type.lower() == 'quad':
        # Total nodes on face: (nGeo+1)**2
        if order == 1:
            return np.arange(4)
        else:
            n           = order
            grid        = np.arange((n+1)**2).reshape(n+1, n+1)
            # Corners: bottom-left, bottom-right, top-right, top-left
            corners     = np.array((grid[0, 0], grid[0, n], grid[n, n], grid[n, 0]))
            # Bottom edge (excluding corners): row 0, columns 1 to n-1 (left-to-right)
            bottom_edge = grid[0, 1:n]
            # Right edge: column n, rows 1 to n-1 (bottom-to-top)
            right_edge  = grid[1:n, n]
            # Top edge: row n, columns n-1 to 1 (right-to-left)
            top_edge    = grid[n, n-1:0:-1]
            # Left edge: column 0, rows n-1 to 1 (top-to-bottom)
            left_edge   = grid[n-1:0:-1, 0]
            # Interior nodes: remaining nodes in row-major order
            interior    = grid[1:n, 1:n].flatten()
            # Assemble ordering: corners, edges, interior
            # ordering    = np.concatenate((corners, bottom_edge, right_edge, top_edge, left_edge, interior))
            ordering    = np.concatenate((corners, bottom_edge, right_edge, top_edge, left_edge, interior))
            return ordering

    elif side_type.lower() == 'triangle':
        # Total nodes on face: (nGeo+1)*(nGeo+2)//2
        if order == 1:
            return np.arange(3)
        else:
            p           = order
            # Build the tensor ordering as a list of (i, j) for which i+j <= p.
            nodes       = []
            for i in range(p+1):
                for j in range(p+1 - i):
                    nodes.append((i, j))
            # Define vertices in the reference triangle:
            vertices    = [(0, 0), (p, 0), (0, p)]
            # Edge from vertex0 (0,0) to vertex1 (p,0): nodes with j==0 (excluding vertices)
            edge01      = [(i, 0) for i in range(1, p)]
            # Edge from vertex1 (p,0) to vertex2 (0,p): nodes on the line i+j==p (excluding vertices)
            edge12      = [(i, p-i) for i in range(p-1, 0, -1)]
            # Edge from vertex2 (0,p) to vertex0 (0,0): nodes with i==0 (excluding vertices)
            edge20      = [(0, j) for j in range(1, p)]
            # Interior nodes: those not on the boundary
            boundary    = set(vertices + edge01 + edge12 + edge20)
            interior    = [node for node in nodes if node not in boundary]
            # Assemble ordering: vertices, then edge nodes in order, then interior nodes.
            desired     = vertices + edge01 + edge12 + edge20 + interior
            ordering    = [nodes.index(nd) for nd in desired]
            return np.array(ordering)
    else:
        raise ValueError(f'Unsupported side type: {side_type}')


@cache
def flip_s2m(N: int, p: int, q: int, flip: int, elemType: Union[str, int], dtype=int) -> np.ndarray:
    """ Transform coordinates from RHS of slave to RHS of master
    """
    flip_map = {  # Tetrahedron
                  # Pyramid
                  # Wedge / Prism
                  # Hexahedron
                  8: {0: np.array((p    ,     q), dtype=dtype),
                      1: np.array((q    ,     p), dtype=dtype),
                      2: np.array((N - p,     q), dtype=dtype),
                      3: np.array((N - q, N - p), dtype=dtype),
                      4: np.array((p    , N - q), dtype=dtype)}
               }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in flip_map:
        raise ValueError(f'Error in flip_s2m: elemType {elemType} is not supported')

    try:
        return flip_map[elemType % 100][flip]
    except KeyError:
        raise KeyError(f'Error in flip_s2m: face {flip} is not supported')


@cache
def cgns_sidetovol(N: int, r: int, p: int, q: int, face: str, elemType: Union[str, int], dtype=int) -> np.ndarray:
    """ Transform coordinates from RHS of side into volume
    """
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: {'x-': np.array((r    , q    , p    ), dtype=dtype),
                       'x+': np.array((N - r, p    , q    ), dtype=dtype),
                       'y-': np.array((p    , r    , q    ), dtype=dtype),
                       'y+': np.array((N - p, N - r, q    ), dtype=dtype),
                       'z-': np.array((q    , p    , r    ), dtype=dtype),
                       'z+': np.array((p    , q    , N - r), dtype=dtype)}
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in cgns_sidetovol: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][face]
    except KeyError:
        raise KeyError(f'Error in cgns_sidetovol: face {face} is not supported')


@cache
def sidetovol2(N: int, flip: int, face: str, elemType: Union[str, int]) -> np.ndarray:
    """ Transform coordinates from RHS of side into volume
    """
    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    # Get the reordering of the element nodes
    mapLin = LINMAP(elemType, order=N)
    # Build the (p,q) grid as arrays of shape (0:N, 0:N)
    P, Q = np.meshgrid(np.arange(N+1, dtype=int),
                       np.arange(N+1, dtype=int), indexing='ij')
    # Build (r) vector for flat surface
    R    = np.zeros_like(P, dtype=int)
    # Vectorize flip_s2m to get the flipped (p, q) values
    vec_flip = (np.vectorize(lambda p, q: flip_s2m(N, p, q, flip, elemType)[0], otypes=[int]),
                np.vectorize(lambda p, q: flip_s2m(N, p, q, flip, elemType)[1], otypes=[int]))
    pq       = tuple([vec_flip[s](P, Q) for s in (0, 1)])
    # Vectorize the cgns_sidetovol function
    vec_cgns =  np.vectorize(lambda r, p, q: cgns_sidetovol(N, r, int(p), int(q), face, elemType), otypes=[int],
                                    signature='(),(),()->(n)')
    # idx_arr will have shape (0:N, 0:N, 3)
    idx_arr = vec_cgns(R, pq[0], pq[1])
    # Use the computed indices from idx_arr to index mapLin
    map = mapLin[idx_arr[..., 0], idx_arr[..., 1], idx_arr[..., 2]]
    return map


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
        elemType = elemTypeClass.name[elemType]

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
        elemType = elemTypeClass.name[elemType]

    order     = nGeo
    # https://hopr.readthedocs.io/en/latest/_images/CGNS_edges.jpg
    # faces_map = {  # Tetrahedron
    #                4: {  # Sides aligned with the axes
    #                      'z-': [s for s  in LINMAP(104 if order == 1 else 204, order=order)[:    , :    , 0    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'y-': [s for s  in LINMAP(104 if order == 1 else 204, order=order)[:    , 0    , :    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'x-': [s for s  in LINMAP(104 if order == 1 else 204, order=order)[0    , :    , :    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      # Diagonal side
    #                      'x+': [s for i in range(order+1)
    #                               for j in range(order+1-i) for s in [LINMAP(104 if order == 1 else 204, order=order)[i, j, order-i-j]] if s != -1]},  # noqa: E272, E501
    #                # Pyramid
    #                5: {  # Sides aligned with the axes
    #                      'z-': [s for s  in LINMAP(105 if order == 1 else 205, order=order)[:    , :    , 0    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'y-': [s for s  in LINMAP(105 if order == 1 else 205, order=order)[:    , 0    , :    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'x-': [s for s  in LINMAP(105 if order == 1 else 205, order=order)[0    , :    , :    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      # Diagonal sides
    #                      'x+': [s for i  in range(order+1) for s in LINMAP(105 if order == 1 else 205, order=order)[:, order-i, i] if s != -1],   # noqa: E272, E501
    #                      'y+': [s for i  in range(order+1) for s in LINMAP(105 if order == 1 else 205, order=order)[order-i, :, i] if s != -1]},  # noqa: E272, E501
    #                # Wedge
    #                6: {  # Sides aligned with the axes
    #                      'y-': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[:    , 0    , :    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'x-': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[0    , :    , :    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'z-': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[:    , :    , 0    ].flatten()         if s != -1],   # noqa: E272, E501
    #                      'z+': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[:    , :    , order].flatten()         if s != -1],   # noqa: E272, E501
    #                      # Diagonal side
    #                      'x+': [s for i  in range(order+1) for s in LINMAP(106 if order == 1 else 206, order=order)[i, order-i, :] if s != -1]},  # noqa: E272, E501
    #                # Hexahedron
    #                8: {  'z-':              LINMAP(108 if order == 1 else 208, order=order)[:    , :    , 0    ] ,                                # noqa: E272, E501
    #                      'y-': np.transpose(LINMAP(108 if order == 1 else 208, order=order)[:    , 0    , :    ]),                                # noqa: E272, E501
    #                      'x+': np.transpose(LINMAP(108 if order == 1 else 208, order=order)[order, :    , :    ]),                                # noqa: E272, E501
    #                      'y+':              LINMAP(108 if order == 1 else 208, order=order)[:    , order, :    ] ,                                # noqa: E272, E501
    #                      'x-':              LINMAP(108 if order == 1 else 208, order=order)[0    , :    , :    ] ,                                # noqa: E272, E501
    #                      'z+': np.transpose(LINMAP(108 if order == 1 else 208, order=order)[:    , :    , order])}                                # noqa: E272, E501
    #
    #             }
    # if elemType % 100 not in faces_map:
    #     raise ValueError(f'Error in face_to_nodes: elemType {elemType} is not supported')
    #
    # try:
    #     return faces_map[elemType % 100][face]
    # except KeyError:
    #     raise KeyError(f'Error in face_to_cgns: face {face} is not supported')

    match elemType % 100:
        case 4:  # Tetrahedron
            faces_map = {  # Sides aligned with the axes
                           'z-': [s for s  in LINMAP(104 if order == 1 else 204, order=order)[:    , :    , 0    ].flatten()         if s != -1],   # noqa: E272, E501
                           'y-': [s for s  in LINMAP(104 if order == 1 else 204, order=order)[:    , 0    , :    ].flatten()         if s != -1],   # noqa: E272, E501
                           'x-': [s for s  in LINMAP(104 if order == 1 else 204, order=order)[0    , :    , :    ].flatten()         if s != -1],   # noqa: E272, E501
                           # Diagonal side
                           'x+': [s for i in range(order+1)
                                    for j in range(order+1-i) for s in [LINMAP(104 if order == 1 else 204, order=order)[i, j, order-i-j]] if s != -1]    # noqa: E272, E501
                        }
        case 5:  # Pyramid
            faces_map = {  # Sides aligned with the axes
                           'z-': [s for s  in LINMAP(105 if order == 1 else 205, order=order)[:    , :    , 0    ].flatten()         if s != -1],   # noqa: E272, E501
                           'y-': [s for s  in LINMAP(105 if order == 1 else 205, order=order)[:    , 0    , :    ].flatten()         if s != -1],   # noqa: E272, E501
                           'x-': [s for s  in LINMAP(105 if order == 1 else 205, order=order)[0    , :    , :    ].flatten()         if s != -1],   # noqa: E272, E501
                           # Diagonal sides
                           'y+': [s for i  in range(order+1) for s in LINMAP(105 if order == 1 else 205, order=order)[:, order-i, i] if s != -1],   # noqa: E272, E501
                           'x+': [s for i  in range(order+1) for s in LINMAP(105 if order == 1 else 205, order=order)[order-i, :, i] if s != -1]    # noqa: E272, E501
                        }
        case 6:  # Wedge
            faces_map = {  # Sides aligned with the axes
                           'y-': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[:    , 0    , :    ].flatten()         if s != -1],   # noqa: E272, E501
                           'x-': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[0    , :    , :    ].flatten()         if s != -1],   # noqa: E272, E501
                           'z-': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[:    , :    , 0    ].flatten()         if s != -1],   # noqa: E272, E501
                           'z+': [s for s  in LINMAP(106 if order == 1 else 206, order=order)[:    , :    , order].flatten()         if s != -1],   # noqa: E272, E501
                           # Diagonal side
                           'x+': [s for i  in range(order+1) for s in LINMAP(106 if order == 1 else 206, order=order)[i, order-i, :] if s != -1]    # noqa: E272, E501
                        }
        case 8:  # Hexahedron
            faces_map = {  'z-':              LINMAP(108 if order == 1 else 208, order=order)[:    , :    , 0    ] ,                                # noqa: E272, E501
                           'y-': np.transpose(LINMAP(108 if order == 1 else 208, order=order)[:    , 0    , :    ]),                                # noqa: E272, E501
                           'x+': np.transpose(LINMAP(108 if order == 1 else 208, order=order)[order, :    , :    ]),                                # noqa: E272, E501
                           'y+':              LINMAP(108 if order == 1 else 208, order=order)[:    , order, :    ] ,                                # noqa: E272, E501
                           'x-':              LINMAP(108 if order == 1 else 208, order=order)[0    , :    , :    ] ,                                # noqa: E272, E501
                           'z+': np.transpose(LINMAP(108 if order == 1 else 208, order=order)[:    , :    , order])                                 # noqa: E272, E501
                        }
        case _:
            raise ValueError(f'Error in face_to_nodes: elemType {elemType} is not supported')

    try:
        return faces_map[face]
    except KeyError:
        raise KeyError(f'Error in face_to_nodes: face {face} is not supported')


@cache
def dir_to_nodes(dir: str, elemType: Union[str, int], nGeo: int) -> Tuple[Any, bool]:
    """ Returns the tensor-product nodes associated with a face
    """
    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    # FIXME: check for non-hexahedral elements
    order     = nGeo
    faces_map = {  # Tetrahedron
                   4: { 'z-': ((slice(None), slice(None), 0          ), False),   #              elemNodes[:    , :    , 0    ],  # noqa: E262, E501
                        'y-': ((slice(None), 0          , slice(None)), True ),   # np.transpose(elemNodes[:    , 0    , :    ]), # noqa: E262, E501
                        'x+': ((order      , slice(None), slice(None)), True ),   # np.transpose(elemNodes[order, :    , :    ]), # noqa: E262, E501
                        'x-': ((0          , slice(None), slice(None)), False)},  #              elemNodes[0    , :    , :    ]}, # noqa: E262, E501
                   # Pyramid
                   5: { 'z-': ((slice(None), slice(None), 0          ), False),   #              elemNodes[:    , :    , 0    ],  # noqa: E262, E501
                        'y-': ((slice(None), 0          , slice(None)), True ),   # np.transpose(elemNodes[:    , 0    , :    ]), # noqa: E262, E501
                        'x+': ((order      , slice(None), slice(None)), True ),   # np.transpose(elemNodes[order, :    , :    ]), # noqa: E262, E501
                        'y+': ((slice(None), order      , slice(None)), False),   #              elemNodes[:    , order, :    ],  # noqa: E262, E501
                        'x-': ((0          , slice(None), slice(None)), False)},  #              elemNodes[0    , :    , :    ]}, # noqa: E262, E501
                   # Wedge / Prism
                   6: { 'z-': ((slice(None), slice(None), 0          ), False),   #              elemNodes[:    , :    , 0    ],  # noqa: E262, E501
                        'y-': ((slice(None), 0          , slice(None)), True ),   # np.transpose(elemNodes[:    , 0    , :    ]), # noqa: E262, E501
                        'x+': ((order      , slice(None), slice(None)), True ),   # np.transpose(elemNodes[order, :    , :    ]), # noqa: E262, E501
                        'x-': ((0          , slice(None), slice(None)), False),   #              elemNodes[0    , :    , :    ],  # noqa: E262, E501
                        'z+': ((slice(None), slice(None), order      ), True )},  # np.transpose(elemNodes[:    , :    , order])},# noqa: E262, E501
                   # Hexahedron
                   8: { 'z-': ((slice(None), slice(None), 0          ), False),   #              elemNodes[:    , :    , 0    ],  # noqa: E262, E501
                        'y-': ((slice(None), 0          , slice(None)), True ),   # np.transpose(elemNodes[:    , 0    , :    ]), # noqa: E262, E501
                        'x+': ((order      , slice(None), slice(None)), True ),   # np.transpose(elemNodes[order, :    , :    ]), # noqa: E262, E501
                        'y+': ((slice(None), order      , slice(None)), False),   #              elemNodes[:    , order, :    ],  # noqa: E262, E501
                        'x-': ((0          , slice(None), slice(None)), False),   #              elemNodes[0    , :    , :    ],  # noqa: E262, E501
                        'z+': ((slice(None), slice(None), order      ), True )}   # np.transpose(elemNodes[:    , :    , order])} # noqa: E262, E501
                 }
    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in face_to_cgns: elemType {elemType} is not supported')

    try:
        return faces_map[elemType % 100][dir]
    except KeyError:
        raise KeyError(f'Error in face_to_cgns: face {dir} is not supported')


# > Not cacheable, we pass mesh[meshio.Mesh]
def count_elems(mesh: meshio.Mesh) -> int:
    nElems = 0
    for _, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in elemTypeClass.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)
        nElems += ioelems.shape[0]
    return nElems


# > Not cacheable, we pass mesh[meshio.Mesh]
def calc_elem_bary(elems: list) -> np.ndarray:
    """
    Compute barycenters of all three-dimensional elements in the mesh.

    Returns:
        elem_bary (np.ndarray): Array of barycenters for all 3D elements, concatenated.
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import numpy as np
    # ------------------------------------------------------

    elem_bary = []
    for elem in elems:
        # Calculate barycenters
        bary = np.mean(mesh_vars.mesh.points[elem.nodes], axis=0)
        elem_bary.append(bary)

    return np.asarray(elem_bary)


@cache
def LINTEN(elemType: int, order: int = 1) -> tuple[np.ndarray, dict[np.int64, int]]:
    """ MESHIO -> IJK ordering for element volume nodes
    """
    # Local imports ----------------------------------------
    # from pyhope.io.formats.cgns import genHEXMAPCGNS
    # from pyhope.io.formats.vtk import genHEXMAPVTK
    from pyhope.io.formats.meshio import TETRMAPMESHIO, PYRAMAPMESHIO, PRISMAPMESHIO, HEXMAPMESHIO
    # ------------------------------------------------------
    # Check if we try to access a curved element with a straight-sided mapping
    if order > 1 and elemType < 200:
        raise ValueError(f'Error in LINTEN: order {order} is not supported for elemType {elemType}')

    match elemType:
        # Straight-sided elements, hard-coded
        case 104:  # Tetraeder
            # return np.array((0, 1, 2, 3))
            TETRTEN = np.array((0, 1, 2, 3))
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENTETR   = {k: v for v, k in enumerate(TETRTEN)}
            return TETRTEN, TENTETR
        case 105:  # Pyramid
            # return np.array((0, 1, 3, 2, 4))
            PYRATEN = np.array((0, 1, 3, 2, 4))
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENPYRA   = {k: v for v, k in enumerate(PYRATEN)}
            return PYRATEN, TENPYRA
        case 106:  # Prism
            # return np.array((0, 1, 2, 3, 4, 5))
            PRISTEN = np.array((0, 1, 2, 3, 4, 5))
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENPRIS   = {k: v for v, k in enumerate(PRISTEN)}
            return PRISTEN, TENPRIS
        case 108:  # Hexaeder
            # return np.array((0, 1, 3, 2, 4, 5, 7, 6))
            HEXTEN = np.array((0, 1, 3, 2, 4, 5, 7, 6))
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENHEX    = {k: v for v, k in enumerate(HEXTEN)}
            return HEXTEN, TENHEX
        # Curved elements, use mapping
        case 204:  # Tetraeder
            _, TETRTEN = TETRMAPMESHIO(order+1)
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENTETR   = {k: v for v, k in enumerate(TETRTEN)}
            return TETRTEN, TENTETR
        case 205:  # Pyramid
            _, PYRATEN = PYRAMAPMESHIO(order+1)
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENPYRA   = {k: v for v, k in enumerate(PYRATEN)}
            return PYRATEN, TENPYRA
        case 206:  # Prism
            _, PRISTEN = PRISMAPMESHIO(order+1)
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENPRIS   = {k: v for v, k in enumerate(PRISTEN)}
            return PRISTEN, TENPRIS
        case 208:  # Hexaeder
            # > HEXTEN : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
            # > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)

            # # CGNS
            # _, HEXTEN = HEXMAPCGNS(order+1)

            # # VTK
            # _, HEXTEN = HEXMAPVTK(order+1)

            # MESHIO
            _, HEXTEN = HEXMAPMESHIO(order+1)
            # meshio accesses them in their own ordering
            # > need to reverse the mapping
            TENHEX    = {k: v for v, k in enumerate(HEXTEN)}
            return HEXTEN, TENHEX
        case _:  # Default
            print('Error in LINTEN, unknown elemType')
            sys.exit(1)


@cache
def LINMAP(elemType: int, order: int = 1) -> npt.NDArray[np.int32]:
    """ MESHIO -> IJK ordering for element corner nodes
    """
    # Local imports ----------------------------------------
    # from pyhope.io.formats.cgns import HEXMAPCGNS
    # from pyhope.io.formats.vtk import HEXMAPVTK
    from pyhope.io.formats.meshio import TETRMAPMESHIO, PYRAMAPMESHIO, PRISMAPMESHIO, HEXMAPMESHIO
    # ------------------------------------------------------
    # Check if we try to access a curved element with a straight-sided mapping
    if order > 1 and elemType < 200:
        raise ValueError(f'Error in LINTEN: order {order} is not supported for elemType {elemType}')

    match elemType:
        # Straight-sided elements, hard-coded
        case 104:  # Tetraeder
            linmap = np.full((2, 2, 2), -1, dtype=np.int32)
            indices = [ (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
            for i, index in enumerate(indices):
                linmap[index] = i
            return linmap
        case 105:  # Pyramid
            linmap = np.full((2, 2, 2), -1, dtype=np.int32)
            indices = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                        (0, 0, 1)]
            for i, index in enumerate(indices):
                linmap[index] = i
            return linmap
        case 106:  # Prism
            linmap = np.full((2, 2, 2), -1, dtype=np.int32)
            indices = [ (0, 0, 0), (1, 0, 0), (0, 1, 0),
                        (0, 0, 1), (1, 0, 1), (0, 1, 1)]
            for i, index in enumerate(indices):
                linmap[index] = i
            return linmap
        case 108:  # Hexaeder
            linmap = np.zeros((2, 2, 2), dtype=np.int32)
            indices = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1) ]
            for i, index in enumerate(indices):
                linmap[index] = i
            return linmap

        # Curved elements, use mapping
        case 204:  # Tetraeder
            TETRMAP, _ = TETRMAPMESHIO(order+1)
            return TETRMAP
        case 205:  # Pyramid
            PYRAMAP, _ = PYRAMAPMESHIO(order+1)
            return PYRAMAP
        case 206:  # Prism
            PRISMAP, _ = PRISMAPMESHIO(order+1)
            return PRISMAP
        case 208:  # Hexaeder
            # > HEXTEN : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
            # > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)

            # # CGNS
            # HEXMAP  , _ = HEXMAPCGNS(order+1)

            # # VTK
            # HEXMAP  , _ = HEXMAPVTK(order+1)

            # MESHIO
            HEXMAP  , _ = HEXMAPMESHIO(order+1)
            return HEXMAP
        case _:  # Default
            print('Error in LINMAP, unknown elemType')
            sys.exit(1)
