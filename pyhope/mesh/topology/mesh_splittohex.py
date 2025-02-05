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
import traceback
from functools import cache
from itertools import chain
from typing import Tuple
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


def MeshSplitToHex(mesh: meshio.Mesh) -> meshio.Mesh:
    """ Split simplex elements into hexahedral elements

        > This routine is mostly identical to MeshChangeElemType
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_vars import nGeo
    from pyhope.readintools.readintools import GetLogical, GetIntFromStr, CountOption
    # ------------------------------------------------------

    if CountOption('doSplitToHex') == 0:
        hopout.sep()
        return mesh

    hopout.separator()
    hopout.info('SPLITTING ELEMENTS TO HEXAHEDRA...')
    hopout.sep()

    splitToHex = GetLogical('doSplitToHex')
    if not splitToHex:
        hopout.separator()

    # Sanity check
    # > Check if all requested element types are hexahedral
    nElemTypes = CountOption('ElemType')
    for iElemType in range(nElemTypes):
        elemType = GetIntFromStr('ElemType', number=iElemType)

        if elemType % 100 != 8:
            # Simplex elements requested
            hopout.warning('Non-hexahedral elements are not supported for splitting, exiting...')

    # > Check if the requested polynomial order is 1
    if nGeo > 1:
        hopout.warning('nGeo = {} not supported for element splitting'.format(nGeo))
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    # Copy original points
    points    = mesh.points.copy()
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    # Prepare new cell blocks and new cell_sets
    elems_new = {}
    csets_new = {}

    # Convert the (triangle/quad) boundary cell set into a dictionary
    csets_old = {}

    # Calculate the offset for the quad cells
    offset    = 0
    for elems in elems_old:
        if any(sub in elems.type for sub in {'vertex', 'line'}):
            offset += len(elems.data)

    for cname, cblock in cell_sets.items():
        if cblock is None:
            continue

        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID, block in enumerate(cblock):
            if elems_old[blockID].type[:4] != 'quad' and elems_old[blockID].type[:8] != 'triangle':
                continue

            if block is None:
                continue

            # Sort them as a set for membership checks
            for face in block:
                nodes = mesh.cells_dict[elems_old[blockID].type][face - offset]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    nPoints  = len(points)
    nFaces   = np.zeros(2)
    faceNum  = 0  # We only build quad faces
    faceType = ['quad', 'hexahedron']
    subNodes = hexa_faces()

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        if ctype[:10] == 'hexahedron':
            continue

        elemSplitter = { 'tetra': (tet_to_hex_points  , tet_to_hex_split  , tet_to_hex_faces  ),
                         'wedge': (prism_to_hex_points, prism_to_hex_split, prism_to_hex_faces)}
        splitPoints, splitElems, splitFaces = elemSplitter.get(ctype, (None, None, None))

        # Only process valid splits
        if splitPoints is None or splitElems is None or splitFaces is None:
            continue

        # Setup split functions
        subIdxs            = splitElems()
        oldFIdxs, subFIdxs = splitFaces()

        # Split each element
        for elem in cdata:
            # Create array for new nodes
            points, newNodes, nPoints = splitPoints(elem, points, nPoints)

            # Reconstruct the cell sets for the boundary conditions
            # > They already exists for the triangular faces, but we create new quad faces with the edge and face centers
            # oldFaces, newFaces  = splitFaces(newNodes)
            oldFaces = [frozenset(newNodes[oldFIdx]) for oldFIdx in oldFIdxs]
            newFaces = [          newNodes[subFIdx]  for subFIdx in subFIdxs]  # noqa: E272

            for oldFace, subFaces in zip(oldFaces, newFaces):
                # Check if the triangular faces is a boundary face
                for cnodes, cname in csets_old.items():
                    cname = ''.join(list(chain.from_iterable(cname)))
                    if oldFace.issubset(cnodes):
                        # Create the new quadrilateral boundary faces
                        for subFace in subFaces:
                            csets_old.setdefault(frozenset(subFace), []).append(cname)
                        # Done with this triangular face, break the (inner) loop
                        break

            subElems = [newNodes[subIdx] for subIdx in subIdxs]

            for subElem in subElems:
                # Assemble the 6 hexahedral faces
                for subNode in subNodes:
                    subFace = subElem[subNode]
                    faceSet = frozenset(subFace)

                    for cnodes, cname in csets_old.items():
                        # Face is not a subset of an existing boundary face
                        if not faceSet.issubset(cnodes):
                            continue

                        # For the first side on the BC, the dict does not exist
                        try:
                            prevSides          = csets_new[cname[0]]
                            prevSides[faceNum] = np.append(prevSides[faceNum], nFaces[faceNum]).astype(int)
                        except KeyError:
                            # We only create the 2D and 3D elements
                            prevSides          = [np.array([], dtype=int) for _ in range(2)]
                            prevSides[faceNum] = np.asarray([nFaces[faceNum]]).astype(int)
                            csets_new.update({cname[0]: prevSides})

                    try:
                        elems_new[faceType[faceNum]] = np.append(elems_new[faceType[faceNum]], np.array([subFace]).astype(int), axis=0)  # noqa: E501
                    except KeyError:
                        elems_new[faceType[faceNum]] = np.array([subFace]).astype(int)

                    nFaces[faceNum] += 1

            try:
                elems_new['hexahedron'] = np.append(elems_new['hexahedron'], np.array(subElems).astype(int), axis=0)  # noqa: E501
            except KeyError:
                elems_new['hexahedron'] = np.array(subElems).astype(int)

    mesh = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251
    hopout.separator()

    return mesh


@cache
def hexa_faces() -> list[np.ndarray]:
    """ Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
        return a list of new hexahedral face connectivity lists.
    """
    return [np.array([0, 1, 2, 3], dtype=int),
            np.array([4, 5, 6, 7], dtype=int),
            np.array([0, 1, 5, 4], dtype=int),
            np.array([2, 3, 7, 6], dtype=int),
            np.array([0, 3, 7, 4], dtype=int),
            np.array([1, 2, 6, 5], dtype=int),
           ]


def tet_to_hex_points(elem: np.ndarray, points: np.ndarray, nPoints: int) -> Tuple[np.ndarray, np.ndarray, int]:
    # Create an array for the new nodes
    newPoints = np.zeros((11, 3), dtype=np.float64)
    # Corner nodes
    # newPoints[:4] = points[elem]
    # Nodes on edges
    newPoints[ 0] = np.mean(points[elem[[0, 1   ]]], axis=0)  # index 4
    newPoints[ 1] = np.mean(points[elem[[1, 2   ]]], axis=0)  # index 5
    newPoints[ 2] = np.mean(points[elem[[0, 2   ]]], axis=0)  # index 6
    newPoints[ 3] = np.mean(points[elem[[0, 3   ]]], axis=0)  # index 7
    newPoints[ 4] = np.mean(points[elem[[1, 3   ]]], axis=0)  # index 8
    newPoints[ 5] = np.mean(points[elem[[2, 3   ]]], axis=0)  # index 9
    # Nodes on faces
    newPoints[ 6] = np.mean(points[elem[[0, 1, 2]]], axis=0)  # index 10
    newPoints[ 7] = np.mean(points[elem[[0, 1, 3]]], axis=0)  # index 11
    newPoints[ 8] = np.mean(points[elem[[1, 2, 3]]], axis=0)  # index 12
    newPoints[ 9] = np.mean(points[elem[[0, 2, 3]]], axis=0)  # index 13
    # Node inside
    newPoints[10] = np.mean(points[elem[:        ]], axis=0)  # index 14
    points = np.append(points, newPoints, axis=0)

    # Assemble list of all the nodes
    # newNodes = list(elem) + np.arange(nPoints, nPoints + 11).tolist()
    newNodes = np.concatenate((elem, np.arange(nPoints, nPoints + 11)))
    nPoints += 11

    return points, newNodes, nPoints


@cache
def tet_to_hex_faces() -> Tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return the 4 triangular faces and the 12 quadrilateral faces.
    """
    # Triangular faces
    oldFaces  = [  np.array([  0,  1,  2    ], dtype=int),
                   np.array([  0,  2,  3    ], dtype=int),
                   np.array([  0,  3,  1    ], dtype=int),
                   np.array([  1,  2,  3    ], dtype=int)
                ]

    # Quadrilateral faces
    newFaces  = [  # First triangle
                 [ np.array([  0,  4,  6, 10], dtype=int),
                   np.array([  4,  1,  5, 10], dtype=int),
                   np.array([  5,  2,  6, 10], dtype=int)],
                   # Second triangle
                 [ np.array([  0,  6,  7, 13], dtype=int),
                   np.array([  6,  2,  9, 13], dtype=int),
                   np.array([  9,  3,  7, 13], dtype=int)],
                   # Third triangle
                 [ np.array([  0,  4,  7, 11], dtype=int),
                   np.array([  4,  1,  8, 11], dtype=int),
                   np.array([  8,  3,  7, 11], dtype=int)],
                   # Fourth triangle
                 [ np.array([  1,  5,  8, 12], dtype=int),
                   np.array([  5,  2,  9, 12], dtype=int),
                   np.array([  9,  3,  8, 12], dtype=int)]
                ]

    return oldFaces, newFaces


@cache
def tet_to_hex_split() -> list[np.ndarray]:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return a list of new hexahedral element connectivity lists.
    """
    return [np.array([  0,  4, 10,  6,  7, 11, 14, 13], dtype=int),
            np.array([  1,  5, 10,  4,  8, 12, 14, 11], dtype=int),
            np.array([  2,  6, 10,  5,  9, 13, 14, 12], dtype=int),
            np.array([  3,  7, 13,  9,  8, 11, 14, 12], dtype=int),
           ]


def prism_to_hex_points(elem: np.ndarray, points: np.ndarray, nPoints: int) -> Tuple[np.ndarray, np.ndarray, int]:
    # Create an array for the new nodes
    newPoints = np.zeros((8, 3), dtype=np.float64)
    # Corner nodes
    # newPoints[:4] = points[elem]
    # Nodes on edges
    newPoints[ 0] = np.mean(points[elem[[0, 1   ]]], axis=0)  # index 6
    newPoints[ 1] = np.mean(points[elem[[1, 2   ]]], axis=0)  # index 7
    newPoints[ 2] = np.mean(points[elem[[0, 2   ]]], axis=0)  # index 8
    newPoints[ 3] = np.mean(points[elem[[3, 4   ]]], axis=0)  # index 9
    newPoints[ 4] = np.mean(points[elem[[4, 5   ]]], axis=0)  # index 10
    newPoints[ 5] = np.mean(points[elem[[3, 5   ]]], axis=0)  # index 11
    # Nodes on faces
    newPoints[ 6] = np.mean(points[elem[[0, 1, 2]]], axis=0)  # index 12
    newPoints[ 7] = np.mean(points[elem[[3, 4, 5]]], axis=0)  # index 13
    points = np.append(points, newPoints, axis=0)

    # Assemble list of all the nodes
    newNodes = np.array(list(elem) + np.arange(nPoints, nPoints + 8).tolist())
    nPoints += 8

    return points, newNodes, nPoints


@cache
def prism_to_hex_faces() -> Tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """ Given the 6 corner node indices of a single prism element (indexed 0..5),
        return the 6 -> new <- quadrilateral faces.
    """
    # Faces
    oldFaces  = [  # Triangular faces
                   np.array([  0,  1,  2    ], dtype=int),
                   np.array([  3,  4,  5    ], dtype=int),
                   # Quadrilateral faces
                   np.array([  0,  1,  4,  3], dtype=int),
                   np.array([  1,  2,  5,  4], dtype=int),
                   np.array([  2,  0,  3,  5], dtype=int),
                ]

    # Quadrilateral faces
    newFaces  = [  # First triangle
                 [ np.array([  0,  6, 12,  8], dtype=int),
                   np.array([  1,  7, 12,  6], dtype=int),
                   np.array([  2,  8, 12,  7], dtype=int)],
                   # Second triangle
                 [ np.array([  3,  9, 13, 11], dtype=int),
                   np.array([  4, 10, 13,  9], dtype=int),
                   np.array([  5, 11, 13, 10], dtype=int)],
                   # First quad face
                 [ np.array([  0,  6,  9,  3], dtype=int),
                   np.array([  6,  1,  4,  9], dtype=int)],
                   # Second quad face
                 [ np.array([  1,  7, 10,  4], dtype=int),
                   np.array([  7,  2,  5, 10], dtype=int)],
                   # Third quad face
                 [ np.array([  0,  8, 11,  3], dtype=int),
                   np.array([  8,  2,  5, 11], dtype=int)]
                ]

    return oldFaces, newFaces


@cache
def prism_to_hex_split() -> list[np.ndarray]:
    """ Given the 6 corner node indices of a single prism element (indexed 0..5),
        return a list of new hexahedral element connectivity lists.
    """
    return [np.array([  0,  6, 12,  8,  3,  9, 13, 11], dtype=int),
            np.array([  1,  7, 12,  6,  4, 10, 13,  9], dtype=int),
            np.array([  2,  8, 12,  7,  5, 11, 13, 10], dtype=int),
           ]
