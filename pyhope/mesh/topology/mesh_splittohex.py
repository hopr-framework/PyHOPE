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
from __future__ import annotations
import gc
from collections import defaultdict
from functools import cache
from typing import Final, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    import numpy.typing as npt
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
    from pyhope.common.common_progress import ProgressBar
    from pyhope.mesh.mesh_vars import nGeo
    from pyhope.readintools.readintools import CreateLogical, GetLogical, CountOption
    # ------------------------------------------------------

    # Create logical here to avoid it showing up in the help
    CreateLogical('SplitToHex', multiple=False, default=False)

    if CountOption('doSplitToHex') == 0 and CountOption('SplitToHex') == 0:
        return mesh

    hopout.separator()
    hopout.info('SPLITTING ELEMENTS TO HEXAHEDRA...')
    hopout.sep()

    splitToHex = (GetLogical('doSplitToHex') if CountOption('doSplitToHex') > 0 else False) or \
                 (GetLogical(  'SplitToHex') if CountOption(  'SplitToHex') > 0 else False)
    if not splitToHex:
        hopout.separator()
        return mesh

    # Native meshio data
    cdict: Final[dict] = mesh.cells_dict
    csets: Final[dict] = getattr(mesh, 'cell_sets', {})

    # Copy original points
    # points    = mesh.points.copy()
    points    = mesh.points
    pointl    = cast(list, mesh.points.tolist())
    elems_old = mesh.cells.copy()

    # Sanity check
    # > Check if the requested polynomial order is 1
    if nGeo > 1:
        hopout.error(f'nGeo = {nGeo} not supported for element splitting', traceback=True)

    # > Check if the mesh contains any pyramids
    unsElems: Final[tuple] = ('pyramid')
    if any(s.startswith(x) for x in unsElems for s in cdict):
        unsupported = [s for s in cdict if any(s.startswith(x) for x in unsElems)]
        hopout.error('Element type[s] "{}" are not supported for splitting, exiting...'.format(', '.join(unsupported)))

    faceType = ['triangle'  , 'quad'  ]
    faceNum  = [          3 ,       4 ]

    # Convert the (triangle/quad) boundary cell set into a dictionary
    csets_old = defaultdict(list)

    for cname, cblock in csets.items():
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
                nodes = cdict[elems_old[blockID].type][face]
                csets_old[frozenset(nodes)].append(cname)

    # nPoints  = len(points)
    nPoints  = len(pointl)
    nFaces   = np.zeros(2)

    # Prepare new cell blocks and new cell_sets
    elems_lst = {ftype: [] for ftype in faceType}
    csets_lst = {}

    # Hardcode quad element faces
    faceVal  = 1
    subNodes = hexa_faces()

    # Create the element sets
    ignElems:  Final[tuple] = ('vertex', 'line', 'quad', 'triangle', 'pyramid', 'hexahedron')
    meshcells: Final[tuple] = tuple((k, v) for k, v in cdict.items() if not any(x in k for x in ignElems))
    nTotalElems = sum(zdata.shape[0] for _, zdata in meshcells)
    bar = ProgressBar(value=nTotalElems, title='│             Processing Elements', length=33, threshold=1000)

    # Build an inverted index to map each node to all face keys (from csets_old) that contain it
    nodeToFace = defaultdict(set)
    for subFace in csets_old:
        for node in subFace:
            nodeToFace[node].add(subFace)

    elemSplitter = { 'tetra'     : (tet_to_hex_points  , tet_to_hex_split  , tet_to_hex_faces  ),
                     'wedge'     : (prism_to_hex_points, prism_to_hex_split, prism_to_hex_faces),
                     'hexahedron': (hex_to_hex_points  , hex_to_hex_split  , hex_to_hex_faces  ),
                   }

    # z-Splitting
    # > Detect hybrid meshes that contain hexahedra. In this case, we can only work if we have prisms and hexahedra
    volTypes    = tuple(cell.type for cell in mesh.cells if cell.type not in ('vertex', 'line', 'triangle', 'quad'))
    splitToHexZ = bool(len(set(volTypes)) > 1 and any(t == 'hexahedron' for t in volTypes))
    hexBCSet    = {}
    hexBCQuads  = set()
    if splitToHexZ:
        unszElems  = ('tetrahedron', 'pyramid')
        if any(s.startswith(x) for x in unszElems for s in cdict):
            unsupported = [s for s in cdict if any(s.startswith(x) for x in unszElems)]
            hopout.error('Element type[s] "{}" are not supported for z-splitting, exiting...'.format(', '.join(unsupported)))

        # Inquire if elements should be split in z-director or we should build mortar interfaces
        CreateLogical('SplitToHexZ', multiple=False, default=True)
        splitToHexZ = (GetLogical('doSplitToHexZ')                                          ) and \
                      (GetLogical(  'SplitToHexZ') if CountOption(  'SplitToHexZ') else True) > 0

        # Keep ordered quad connectivities
        if not splitToHexZ:
            for ctype, conn in cdict.items():
                if ctype.startswith('quad'):
                    for q in conn:
                        hexBCSet[frozenset(q)] = np.array(q, dtype=int)

    for cell in elems_old:
        ctype, cdata = cell.type, cell.data

        # Carry over the original hexahedra volume cells unchanged
        if ctype.startswith('hexahedron') and not splitToHexZ:
            elems_lst.setdefault('hexahedron', []).extend(cdata)

            # Detect which existing quad boundary faces are attached to these hexahedra
            for elem in cdata:
                for face in hexa_faces():
                    fNodes = np.array(elem)[face]
                    fSet   = frozenset(fNodes)

                    candidate_sets = [nodeToFace[node] for node in fSet if node in nodeToFace]
                    if not candidate_sets:
                        continue

                    common_candidates = set.intersection(*candidate_sets)
                    for candidate in common_candidates:
                        if fSet.issubset(candidate):
                            hexBCQuads.add(candidate)

            # Skip the rest of the hexahedrons
            continue

        splitPoints, splitElems, splitFaces = elemSplitter.get(ctype, (None, None, None))

        # Only process valid splits
        if splitPoints is None or splitElems is None or splitFaces is None:
            continue

        # Setup split functions
        subIdxs            = splitElems(          z_split=splitToHexZ)
        oldFIdxs, subFIdxs = splitFaces(          z_split=splitToHexZ)
        subPts             = splitPoints(order=1, z_split=splitToHexZ)

        # Iterate over element types
        for elem in cdata:
            # Create array for new nodes
            newPoints = tuple(np.mean(points[elem[i]], axis=0) for i in subPts)

            # Assemble the new nodes
            # newNodes   = np.concatenate((elem, np.arange(nPoints, nPoints + len(newPoints))))
            newNodes   = np.array(elem.tolist() + list(range(nPoints, nPoints + len(newPoints))), dtype=int)
            nPoints   += len(newPoints)
            pointl.extend(newPoints)

            # Reconstruct the cell sets for the boundary conditions
            # > They already exists for the triangular faces, but we create new quad faces with the edge and face centers
            # oldFaces, newFaces  = splitFaces(newNodes)
            oldFaces = tuple(frozenset(newNodes[oldFIdx]) for oldFIdx in oldFIdxs)
            newFaces = tuple(          newNodes[subFIdx]  for subFIdx in subFIdxs)  # noqa: E272

            # Deferr update for new boundary faces
            # > Instead of updating csets_old repeatedly, we collect deferred updates
            newBCFaces = []  # List of tuples: (new_face_key, combined_name)
            for oldFace, subFaces in zip(oldFaces, newFaces, strict=True):
                # Use the inverted index to get candidate face keys that might contain oldFace
                candidate_sets = [nodeToFace[node] for node in oldFace if node in nodeToFace]
                if not candidate_sets:
                    continue

                # Intersection of candidate sets reduces the number of checks
                common_candidates = set.intersection(*candidate_sets)
                for candidate in common_candidates:
                    if oldFace.issubset(candidate):
                        # Combine the associated names into one string
                        combined_name = ''.join(csets_old[candidate])
                        # Create the new quadrilateral boundary faces by precomputing the frozensets
                        faceSet = [frozenset(face) for face in subFaces]
                        # This cannot be a tuple, we need to iterate over all the candidate sets
                        newBCFaces.extend((key, combined_name) for key in faceSet)
                        # Done with this triangular face, break out of the (inner) candidate loop
                        break

            # Process sub-elements and record new face keys with their indices
            subElems = tuple(newNodes[subIdx] for subIdx in subIdxs)
            subFaces = []  # List of tuples: (new_face_key, face_index)
            for subElem in subElems:
                # Assemble the 6 hexahedral faces
                for subNode in subNodes:
                    subFace = subElem[subNode]
                    faceSet = frozenset(subFace)
                    subFaces.append((faceSet, nFaces[faceVal]))
                    elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))
                    nFaces[faceVal] += 1

            # Merge deferred updates with new face indices
            for newFace, faceName in newBCFaces:
                for subFace, faceIndex in subFaces:
                    if subFace == newFace:
                        csets_lst.setdefault(faceName, [[], []])
                        csets_lst[faceName][faceVal].append(faceIndex)

            # Hardcode hexahedron elements
            if 'hexahedron' not in elems_lst:
                elems_lst['hexahedron'] = []
            # Append all rows from subElems
            # elems_lst['hexahedron'].extend(np.array(subElems, dtype=int).tolist())
            elems_lst['hexahedron'].extend(subElems)

            # Update the progress bar
            bar.step()

    # Add back the existing quad boundary faces attached to carried-over hexahedra
    if not splitToHexZ and hexBCQuads:
        for qset in hexBCQuads:
            qnodes = hexBCSet.get(qset)

            if qnodes is None:
                continue

            faceIndex = int(nFaces[1])
            elems_lst['quad'].append(np.array(qnodes, dtype=int))
            nFaces[1] += 1

            # Update boundary sets for this quad face
            for name in csets_old.get(qset, []):
                csets_lst.setdefault(name, [[], []])
                csets_lst[name][1].append(faceIndex)

    # Close the progress bar
    bar.close()

    # Convert lists to NumPy arrays for elems_new and csets_new
    elems_new = {}
    csets_new = {}

    for key in elems_lst:  # noqa: PLC0206
        if   isinstance(elems_lst[key], list) and     elems_lst[key]:  # noqa: E271
            # Convert the list of accumulated arrays/lists into a single NumPy array
            elems_new[key] = np.array(elems_lst[key], dtype=int)
        elif isinstance(elems_lst[key], list) and not elems_lst[key]:
            # Determine the expected number of columns
            elems_new[key] = np.empty((0, faceNum[faceType.index(key)]), dtype=int)

    for key in csets_lst:  # noqa: PLC0206
        csets_new[key] = tuple(np.array(lst, dtype=int) for lst in csets_lst[key])

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    mesh = meshio.Mesh(points    = points,     # noqa: E251
                       cells     = elems_new,  # noqa: E251
                       cell_sets = csets_new)  # noqa: E251

    # Run garbage collector to release memory
    gc.collect()

    hopout.separator()
    return mesh


@cache
def hexa_faces() -> tuple[npt.NDArray, ...]:
    """ Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
        return a list of new hexahedral face connectivity lists.
    """
    return (np.array((0, 1, 2, 3), dtype=int),
            np.array((4, 5, 6, 7), dtype=int),
            np.array((0, 1, 5, 4), dtype=int),
            np.array((2, 3, 7, 6), dtype=int),
            np.array((0, 3, 7, 4), dtype=int),
            np.array((1, 2, 6, 5), dtype=int),
           )


@cache
def tet_to_hex_points(order: int, z_split: bool = False) -> tuple[npt.NDArray, ...]:
    """ Given the 4 corner nodes of a single tetrahedron element (indexed 0..3),
        return the 10 additional points after splitting
    """
    match order:
        case 1:
            return (  # Nodes on edges
                      np.array((  0,  1    ), dtype=int),           # index 4
                      np.array((  1,  2    ), dtype=int),           # index 5
                      np.array((  0,  2    ), dtype=int),           # index 6
                      np.array((  0,  3    ), dtype=int),           # index 7
                      np.array((  1,  3    ), dtype=int),           # index 8
                      np.array((  2,  3    ), dtype=int),           # index 9
                      # Nodes on faces
                      np.array((  0,  1,  2), dtype=int),           # index 10
                      np.array((  0,  1,  3), dtype=int),           # index 11
                      np.array((  1,  2,  3), dtype=int),           # index 12
                      np.array((  0,  2,  3), dtype=int),           # index 13
                      # Inside node
                      np.arange(  0,  4     , dtype=int)            # index 14
                   )
        case _:
            import pyhope.output.output as hopout
            hopout.error(f'Order {order} not supported for element splitting')


@cache
def tet_to_hex_faces(z_split: bool = False) -> tuple[list[npt.NDArray], list[list[npt.NDArray]]]:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return the 4 triangular faces and the 12 quadrilateral faces
    """
    # Triangular faces
    oldFaces  = [  np.array((  0,  1,  2    ), dtype=int),
                   np.array((  0,  2,  3    ), dtype=int),
                   np.array((  0,  3,  1    ), dtype=int),
                   np.array((  1,  2,  3    ), dtype=int)
                ]

    # Quadrilateral faces
    newFaces  = [  # First triangle
                 [ np.array((  0,  4,  6, 10), dtype=int),
                   np.array((  4,  1,  5, 10), dtype=int),
                   np.array((  5,  2,  6, 10), dtype=int)],
                   # Second triangle
                 [ np.array((  0,  6,  7, 13), dtype=int),
                   np.array((  6,  2,  9, 13), dtype=int),
                   np.array((  9,  3,  7, 13), dtype=int)],
                   # Third triangle
                 [ np.array((  0,  4,  7, 11), dtype=int),
                   np.array((  4,  1,  8, 11), dtype=int),
                   np.array((  8,  3,  7, 11), dtype=int)],
                   # Fourth triangle
                 [ np.array((  1,  5,  8, 12), dtype=int),
                   np.array((  5,  2,  9, 12), dtype=int),
                   np.array((  9,  3,  8, 12), dtype=int)]
                ]

    return oldFaces, newFaces


@cache
def tet_to_hex_split(z_split: bool = False) -> list[npt.NDArray]:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return a list of new hexahedral element connectivity lists
    """
    return [np.array((  0,  4, 10,  6,  7, 11, 14, 13), dtype=int),
            np.array((  1,  5, 10,  4,  8, 12, 14, 11), dtype=int),
            np.array((  2,  6, 10,  5,  9, 13, 14, 12), dtype=int),
            np.array((  3,  7, 13,  9,  8, 11, 14, 12), dtype=int),
           ]


@cache
def prism_to_hex_points(order: int, z_split: bool = False) -> tuple[npt.NDArray, ...]:
    """ Given the 6 corner nodes of a single prism element (indexed 0..6),
        return the 8 additional points after splitting
    """
    match order:
        case 1:
            if not z_split:
                return (  # Nodes on edges
                          np.array((  0,  1    ), dtype=int),           # index 6
                          np.array((  1,  2    ), dtype=int),           # index 7
                          np.array((  0,  2    ), dtype=int),           # index 8
                          np.array((  3,  4    ), dtype=int),           # index 9
                          np.array((  4,  5    ), dtype=int),           # index 10
                          np.array((  3,  5    ), dtype=int),           # index 11
                          # Nodes on faces
                          np.array((  0,  1,  2), dtype=int),           # index 12
                          np.array((  3,  4,  5), dtype=int),           # index 13
                       )
            else:
                # z_split: add vertical mid-edges + inside node
                return (  # Nodes on edges
                          np.array((  0,  1    ), dtype=int),           # index 6
                          np.array((  1,  2    ), dtype=int),           # index 7
                          np.array((  0,  2    ), dtype=int),           # index 8
                          np.array((  3,  4    ), dtype=int),           # index 9
                          np.array((  4,  5    ), dtype=int),           # index 10
                          np.array((  3,  5    ), dtype=int),           # index 11
                          # Nodes on faces
                          np.array((  0,  1,  2), dtype=int),           # index 12
                          np.array((  3,  4,  5), dtype=int),           # index 13
                          # Nodes on vertical edges
                          np.array((  0,  3    ), dtype=int),           # index 14
                          np.array((  1,  4    ), dtype=int),           # index 15
                          np.array((  2,  5    ), dtype=int),           # index 16
                          # Inside node
                          np.arange(  0,  6     , dtype=int),           # index 17
                          # Nodes on rectangular face centers
                          np.array((  0,  1,  3,  4), dtype=int),       # index 18
                          np.array((  1,  2,  4,  5), dtype=int),       # index 19
                          np.array((  0,  2,  3,  5), dtype=int),       # index 20
                       )
        case _:
            import pyhope.output.output as hopout
            hopout.error(f'Order {order} not supported for element splitting')


@cache
def prism_to_hex_faces(z_split: bool = False) -> tuple[list[npt.NDArray], list[list[npt.NDArray]]]:
    """ Given the 6 corner node indices of a single prism element (indexed 0..5),
        return the original faces and the corresponding new quadrilateral sub-faces.
    """
    # Faces
    oldFaces  = [  # Triangular faces
                   np.array((  0,  1,  2    ), dtype=int),
                   np.array((  3,  4,  5    ), dtype=int),
                   # Quadrilateral faces
                   np.array((  0,  1,  4,  3), dtype=int),
                   np.array((  1,  2,  5,  4), dtype=int),
                   np.array((  2,  0,  3,  5), dtype=int),
                ]

    if not z_split:
        newFaces  = [  # First triangle
                     [ np.array((  0,  6, 12,  8), dtype=int),
                       np.array((  1,  7, 12,  6), dtype=int),
                       np.array((  2,  8, 12,  7), dtype=int)],
                       # Second triangle
                     [ np.array((  3,  9, 13, 11), dtype=int),
                       np.array((  4, 10, 13,  9), dtype=int),
                       np.array((  5, 11, 13, 10), dtype=int)],
                       # First quad face
                     [ np.array((  0,  6,  9,  3), dtype=int),
                       np.array((  6,  1,  4,  9), dtype=int)],
                       # Second quad face
                     [ np.array((  1,  7, 10,  4), dtype=int),
                       np.array((  7,  2,  5, 10), dtype=int)],
                       # Third quad face
                     [ np.array((  0,  8, 11,  3), dtype=int),
                       np.array((  8,  2,  5, 11), dtype=int)]
                    ]
    else:
        newFaces  = [  # First triangle
                     [ np.array((  0,  6, 12,  8), dtype=int),
                       np.array((  1,  7, 12,  6), dtype=int),
                       np.array((  2,  8, 12,  7), dtype=int)],
                       # Second triangle
                     [ np.array((  3,  9, 13, 11), dtype=int),
                       np.array((  4, 10, 13,  9), dtype=int),
                       np.array((  5, 11, 13, 10), dtype=int)],
                       # First quad face
                     [ np.array((  0,  6, 18, 14), dtype=int),
                       np.array((  6,  1, 15, 18), dtype=int),
                       np.array(( 14, 18,  9,  3), dtype=int),
                       np.array(( 18, 15,  4,  9), dtype=int)],
                       # Second quad face
                     [ np.array((  1,  7, 19, 15), dtype=int),
                       np.array((  7,  2, 16, 19), dtype=int),
                       np.array(( 15, 19, 10,  4), dtype=int),
                       np.array(( 19, 16,  5, 10), dtype=int)],
                       # Third quad face
                     [ np.array((  2,  8, 20, 16), dtype=int),
                       np.array((  8,  0, 14, 20), dtype=int),
                       np.array(( 16, 20, 11,  5), dtype=int),
                       np.array(( 20, 14,  3, 11), dtype=int)]
        ]

    return oldFaces, newFaces


@cache
def prism_to_hex_split(z_split: bool = False) -> tuple[npt.NDArray, ...]:
    """ Given the 6 corner node indices of a single prism element (indexed 0..5),
        return a list of new hexahedral element connectivity lists
    """
    if not z_split:
        return (np.array((  0,  6, 12,  8,  3,  9, 13, 11), dtype=int),
                np.array((  1,  7, 12,  6,  4, 10, 13,  9), dtype=int),
                np.array((  2,  8, 12,  7,  5, 11, 13, 10), dtype=int),
               )
    else:
        # z_split: split each of the 3 base hexahedra into two stacked hexahedra
        return (np.array((  0,  6, 12,  8, 14, 18, 17, 20), dtype=int),
                np.array(( 14, 18, 17, 20,  3,  9, 13, 11), dtype=int),

                np.array((  1,  7, 12,  6, 15, 19, 17, 18), dtype=int),
                np.array(( 15, 19, 17, 18,  4, 10, 13,  9), dtype=int),

                np.array((  2,  8, 12,  7, 16, 20, 17, 19), dtype=int),
                np.array(( 16, 20, 17, 19,  5, 11, 13, 10), dtype=int),
               )


@cache
def hex_to_hex_points(order: int, z_split: bool = False) -> tuple[npt.NDArray, ...]:
    """ Given the 8 corner nodes of a single hexahedron element (indexed 0..7),
        return the 19 additional points after splitting
    """
    match order:
        case 1:
            return (  # Nodes on edges
                      np.array((  0,  1        ), dtype=int),       # index 8
                      np.array((  1,  2        ), dtype=int),       # index 9
                      np.array((  2,  3        ), dtype=int),       # index 10
                      np.array((  3,  0        ), dtype=int),       # index 11

                      np.array((  4,  5        ), dtype=int),       # index 12
                      np.array((  5,  6        ), dtype=int),       # index 13
                      np.array((  6,  7        ), dtype=int),       # index 14
                      np.array((  7,  4        ), dtype=int),       # index 15

                      np.array((  0,  4        ), dtype=int),       # index 16
                      np.array((  1,  5        ), dtype=int),       # index 17
                      np.array((  2,  6        ), dtype=int),       # index 18
                      np.array((  3,  7        ), dtype=int),       # index 19
                      # Nodes on faces
                      np.array((  0,  1,  2,  3), dtype=int),       # index 20
                      np.array((  4,  5,  6,  7), dtype=int),       # index 21
                      np.array((  0,  1,  5,  4), dtype=int),       # index 22
                      np.array((  1,  2,  6,  5), dtype=int),       # index 23
                      np.array((  2,  3,  7,  6), dtype=int),       # index 24
                      np.array((  3,  0,  4,  7), dtype=int),       # index 25
                      # Inside node
                      np.arange(  0,  8         , dtype=int)        # index 26
                   )

        case _:
            import pyhope.output.output as hopout
            hopout.error(f'Order {order} not supported for hexahedron refinement')


@cache
def hex_to_hex_faces(z_split: bool = False) -> tuple[list[npt.NDArray], list[list[npt.NDArray]]]:
    """ Given the 8 corner node indices of a single hexahedron element (indexed 0..7),
        return the 6 quadrilateral faces and the 24 quadrilateral faces after splitting
    """
    # Quadrilateral faces
    oldFaces  = [  np.array((  0,  1,  2,  3), dtype=int),
                   np.array((  4,  5,  6,  7), dtype=int),
                   np.array((  0,  1,  5,  4), dtype=int),
                   np.array((  1,  2,  6,  5), dtype=int),
                   np.array((  2,  3,  7,  6), dtype=int),
                   np.array((  3,  0,  4,  7), dtype=int),
                ]

    # Quadrilateral faces
    newFaces  = [  # Bottom face
                 [ np.array((  0,  8, 20, 11), dtype=int),
                   np.array((  8,  1,  9, 20), dtype=int),
                   np.array(( 20,  9,  2, 10), dtype=int),
                   np.array(( 11, 20, 10,  3), dtype=int)],
                   # Top face
                 [ np.array((  4, 12, 21, 15), dtype=int),
                   np.array(( 12,  5, 13, 21), dtype=int),
                   np.array(( 21, 13,  6, 14), dtype=int),
                   np.array(( 15, 21, 14,  7), dtype=int)],
                   # Side Face
                 [ np.array((  0,  8, 22, 16), dtype=int),
                   np.array((  8,  1, 17, 22), dtype=int),
                   np.array(( 22, 17,  5, 12), dtype=int),
                   np.array(( 16, 22, 12,  4), dtype=int)],
                   # Side Face
                 [ np.array((  1,  9, 23, 17), dtype=int),
                   np.array((  9,  2, 18, 23), dtype=int),
                   np.array(( 23, 18,  6, 13), dtype=int),
                   np.array(( 17, 23, 13,  5), dtype=int)],
                   # Side Face
                 [ np.array((  2, 10, 24, 18), dtype=int),
                   np.array(( 10,  3, 19, 24), dtype=int),
                   np.array(( 24, 19,  7, 14), dtype=int),
                   np.array(( 18, 24, 14,  6), dtype=int)],
                   # Side Face
                 [ np.array((  3, 11, 25, 19), dtype=int),
                   np.array(( 11,  0, 16, 25), dtype=int),
                   np.array(( 25, 16,  4, 15), dtype=int),
                   np.array(( 19, 25, 15,  7), dtype=int)]
                ]

    return oldFaces, newFaces


@cache
def hex_to_hex_split(z_split: bool = False) -> tuple[npt.NDArray, ...]:
    """ Given the 8 corner node indices of a single hexahedron element (indexed 0..7),
        return a list of new hexahedral element connectivity lists
    """
    return (np.array((  0,  8, 20, 11, 16, 22, 26, 25), dtype=int),
            np.array((  8,  1,  9, 20, 22, 17, 23, 26), dtype=int),
            np.array(( 20,  9,  2, 10, 26, 23, 18, 24), dtype=int),
            np.array(( 11, 20, 10,  3, 25, 26, 24, 19), dtype=int),
            np.array(( 16, 22, 26, 25,  4, 12, 21, 15), dtype=int),
            np.array(( 22, 17, 23, 26, 12,  5, 13, 21), dtype=int),
            np.array(( 26, 23, 18, 24, 21, 13,  6, 14), dtype=int),
            np.array(( 25, 26, 24, 19, 15, 21, 14,  7), dtype=int),
           )
