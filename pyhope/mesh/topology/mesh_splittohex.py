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
        return mesh

    hopout.separator()
    hopout.info('SPLITTING ELEMENTS TO HEXAHEDRA...')
    hopout.sep()

    splitToHex = GetLogical('doSplitToHex')
    if not splitToHex:
        hopout.info('SPLITTING ELEMENTS TO HEXAHEDRA DONE!')
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
        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID, block in enumerate(cblock):
            if elems_old[blockID].type[:4] != 'quad' and elems_old[blockID].type[:8] != 'triangle':
                continue

            # Sort them as a set for membership checks
            for face in block:
                nodes = mesh.cells_dict[elems_old[blockID].type][face - offset]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    nPoints  = len(points)
    nFaces   = np.zeros(2)
    faceNum  = 0  # We only build quad faces
    faceType = ['quad', 'hexahedron']

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        if ctype[:10] == 'hexahedron':
            continue

        # Split each element
        match ctype[:10]:
            case 'tetra':
                for elem in cdata:
                    # Create array for new nodes
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
                    newNodes = elem.tolist() + np.arange(nPoints, nPoints + 11).tolist()

                    # Reconstruct the cell sets for the boundary conditions
                    # > They already exists for the triangular faces, but we create new quad faces with the edge and face centers
                    trias, quads = tet_to_quad_faces(newNodes)
                    for tria, quads in zip(trias, quads):
                        # Check if the triangular faces is a boundary face
                        for cnodes, cname in csets_old.items():
                            cname = ''.join(list(chain.from_iterable(cname)))
                            if tria.issubset(cnodes):
                                # Create the new quadrilateral boundary faces
                                for quad in quads:
                                    csets_old.setdefault(frozenset(quad), []).append(cname)
                                # Done with this triangular face, break the (inner) loop
                                break

                    subElems = split_tet_to_hexs(newNodes)
                    nPoints += 11

                    for subElem in subElems:
                        # Assemble the 6 hexahedral faces
                        faces = hexa_faces(subElem)

                        for subFace in faces:
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

    return mesh


def tet_to_quad_faces(nodes: list) -> Tuple[list, list]:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return the 4 triangular faces and the 12 quadrilateral faces.
    """
    elemNodes = np.array(nodes)

    # Triangular faces
    faces     = [frozenset(elemNodes[[0, 1, 2]]),
                 frozenset(elemNodes[[0, 2, 3]]),
                 frozenset(elemNodes[[0, 3, 1]]),
                 frozenset(elemNodes[[1, 2, 3]])
                ]

    # Quadrilateral faces
    subFaces  = [  # First triangle
                 [tuple((elemNodes[ 0], elemNodes[ 4], elemNodes[ 6], elemNodes[10])),
                  tuple((elemNodes[ 4], elemNodes[ 1], elemNodes[ 5], elemNodes[10])),
                  tuple((elemNodes[ 5], elemNodes[ 2], elemNodes[ 6], elemNodes[10]))],
                   # Second triangle
                 [tuple((elemNodes[ 0], elemNodes[ 6], elemNodes[ 7], elemNodes[13])),
                  tuple((elemNodes[ 6], elemNodes[ 2], elemNodes[ 9], elemNodes[13])),
                  tuple((elemNodes[ 9], elemNodes[ 3], elemNodes[ 7], elemNodes[13]))],
                   # Third triangle
                 [tuple((elemNodes[ 0], elemNodes[ 4], elemNodes[ 7], elemNodes[11])),
                  tuple((elemNodes[ 4], elemNodes[ 1], elemNodes[ 8], elemNodes[11])),
                  tuple((elemNodes[ 8], elemNodes[ 3], elemNodes[ 7], elemNodes[11]))],
                   # Fourth triangle
                 [tuple((elemNodes[ 1], elemNodes[ 5], elemNodes[ 8], elemNodes[12])),
                  tuple((elemNodes[ 5], elemNodes[ 2], elemNodes[ 9], elemNodes[12])),
                  tuple((elemNodes[ 9], elemNodes[ 3], elemNodes[ 8], elemNodes[12]))]
                ]

    return faces, subFaces


def hexa_faces(nodes: list) -> list:
    """ Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
        return a list of new hexahedral face connectivity lists.
    """
    return [tuple((nodes[0], nodes[1], nodes[3], nodes[2])),
            tuple((nodes[4], nodes[5], nodes[7], nodes[6])),
            tuple((nodes[0], nodes[1], nodes[4], nodes[5])),
            tuple((nodes[2], nodes[3], nodes[6], nodes[7])),
            tuple((nodes[0], nodes[3], nodes[4], nodes[7])),
            tuple((nodes[1], nodes[2], nodes[5], nodes[6])),
           ]


def split_tet_to_hexs(nodes: list) -> list:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return a list of new hexahedral element connectivity lists.
    """
    return [tuple((nodes[ 0], nodes[ 4], nodes[10], nodes[ 6], nodes[ 7], nodes[11], nodes[14], nodes[13])),
            tuple((nodes[ 1], nodes[ 5], nodes[10], nodes[ 4], nodes[ 8], nodes[12], nodes[14], nodes[11])),
            tuple((nodes[ 2], nodes[ 6], nodes[10], nodes[ 5], nodes[ 9], nodes[13], nodes[14], nodes[12])),
            tuple((nodes[ 3], nodes[ 7], nodes[13], nodes[ 9], nodes[ 8], nodes[11], nodes[14], nodes[12])),
           ]
