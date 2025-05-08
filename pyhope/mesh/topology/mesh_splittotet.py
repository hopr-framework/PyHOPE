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
from collections import defaultdict
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np

from pyhope.readintools.readintools import GetLogical
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def MeshSplitToTet(mesh: meshio.Mesh) -> meshio.Mesh:
    """ Split simplex elements into hexahedral elements

        > This routine is mostly identical to MeshChangeElemType
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.common.common_progress import ProgressBar
    from pyhope.mesh.mesh_vars import nGeo
    # ------------------------------------------------------

    if not GetLogical('doSplitToTet'):  # and nGeo!=2:
        return mesh

    if not any(key.startswith('pyramid') for key in mesh.cells_dict):
        return mesh

    hopout.separator()
    hopout.info('SPLITTING PYRAMIDS TO TETRAHEDRALS...')
    hopout.sep()

    # Sanity check
    # > Check if all requested element types are pyramids
    #  nElemTypes = CountOption('ElemType')
    #  for iElemType in range(nElemTypes):
    #      elemType = GetIntFromStr('ElemType', number=iElemType)
    #
    #      if elemType % 100 != 5:
    #          # Simplex elements requested
    #          hopout.warning('Only supported pyramids, exiting...')

    # Copy original points
    # points    = mesh.points.copy()
    points    = mesh.points
    pointl    = cast(list, mesh.points.tolist())
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    faceType = ['triangle'  , 'quad'  ]
    faceNum  = [          3 ,       4 ]

    # Convert the (triangle/quad) boundary cell set into a dictionary
    csets_old = defaultdict(list)

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
                nodes = mesh.cells_dict[elems_old[blockID].type][face]
                csets_old[frozenset(nodes)].append(cname)

    nFaces    = np.zeros(2)

    # Get base key to distinguish between linear and high-order elements
    faceMaper = {5: lambda x: 0 if x == 0 else 1}
    nFace     = (nGeo+1)*(nGeo+2)/2
    faceMap   = faceMaper.get(5, None)
    # Sanity check
    if faceMap is None:
        sys.exit(1)

    # Prepare new cell blocks and new cell_sets
    elems_lst = {ftype: [] for ftype in faceType}
    csets_lst = {}

    # Hardcode element types
    ElemType = 'tetra'
    ElemType += '' if nGeo == 1 else str(NDOFperElemType('tetra', nGeo))

    # Prepare face-index functions based on the cell type
    faceIdxFuncs = {
        'hexahedron': lambda: hexa_faces( order=nGeo),  # noqa: E272
        'wedge':      lambda: prism_faces(order=nGeo),  # noqa: E272
        'pyramid':    lambda: pyram_faces(order=nGeo),  # noqa: E272
        'tetra':      lambda: tetra_faces(order=nGeo)   # noqa: E272
    }

    # Build an inverted index to map each node to all face keys (from csets_old) that contain it
    # nodeToFace = defaultdict(set)
    # for subFace in csets_old:
    #     for node in subFace:
    #         nodeToFace[node].add(subFace)

    # Build old face indices for each cell type based on mesh.cells_dict
    oldFIdxs = {}
    for key in mesh.cells_dict:
        for etype, func in faceIdxFuncs.items():
            if key.startswith(etype):
                oldFIdxs[key] = func()
                break

    # Build an inverted index to map each node to all face keys (from csets_old) that contain it
    nodeToFace = defaultdict(set)
    for subFace in csets_old:
        for node in subFace:
            nodeToFace[node].add(subFace)

    # Sort out all pyramids
    for cell in elems_old:
        ctype, cdata = cell.type, cell.data

        if ctype.startswith('triangle') or ctype.startswith('quad'):
            continue

        # Iterate over element types
        for elem in cdata:
            # For pyramids that are meant to be split later, skip elements with all first 4 points having y==1 or 2
            if ctype.startswith('pyramid'):
                pts = np.array(points[elem])
                if np.all(pts[:4, 1] == 1.) or np.all(pts[:4, 1] == 2.):
                    continue

            nodes = np.array(elem.tolist(), dtype=int)
            elems_lst.setdefault(ctype, []).append(elem)

            oldFaces = [nodes[oldFIdx] for oldFIdx in oldFIdxs[ctype]]

            # Compute the boundary faces for the element using precomputed face indices
            currentFaceIdxs = oldFIdxs[ctype]
            oldFaces = [nodes[idx] for idx in currentFaceIdxs]

            for subFace in oldFaces:
                faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)
                faceSet = frozenset(subFace)

                # Use the inverted index to efficiently narrow down candidate boundary face definitions.
                candidate_sets = [nodeToFace[node] for node in faceSet if node in nodeToFace]
                if candidate_sets:
                    common_candidates = set.intersection(*candidate_sets)
                    for candidate in common_candidates:
                        if faceSet.issubset(candidate):
                            for name in csets_old[candidate]:
                                csets_lst.setdefault(name, [[], []])
                                csets_lst[name][faceVal].append(nFaces[faceVal])
                elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))
                nFaces[faceVal] += 1

    # Create the element sets
    meshcells = [cell for cell in mesh.cells if cell.type.startswith('pyramid')]
    nTotalElems = sum(cell.data.shape[0] for cell in meshcells)
    bar = ProgressBar(value=nTotalElems, title='│             Processing Elements', length=33, threshold=1000)

    elemSplitter = {'pyramid': (pyram_to_tet_split, pyram_to_tet_faces)}

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        # Only process pyramids for splitting
        if not ctype.startswith('pyramid'):
            continue

        splitElems, splitFaces = elemSplitter.get(ctype, (None, None))

        # Only process valid splits
        if splitElems is None or splitFaces is None:
            continue

        # Setup split functions
        subIdxs  = splitElems(order=nGeo)
        subFIdxs = splitFaces(order=nGeo)

        # Process each element in cell data
        for elem in cdata:
            # Skip elements whose first 4 points do not meet the criteria
            if not (np.all(np.array(points[elem])[:4, 1] == 1.) or np.all(np.array(points[elem])[:4, 1] == 2.)):
                continue

            # Split each element into sub-elements
            subElems = elem[subIdxs]

            # Initialize lists to collect deferred updates and new face indices
            newBCFaces = []   # List of tuples (faceSet, cname, faceVal)
            subFaces   = []   # List of tuples (faceSet, faceIndex) corresponding to new faces

            # Process each sub-element
            for subElem in subElems:
                # The new faces for this sub-element based on subFIdxs
                newFaces = [subElem[face] for face in subFIdxs]

                for subFace in newFaces:
                    # Determine face type based on length criteria
                    faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)
                    faceSet = frozenset(subFace)
                    # Record the current index of the new face
                    currentFaceIndex = nFaces[faceVal]
                    # Save the face key and its index for later deferred update merging
                    subFaces.append((faceSet, currentFaceIndex))

                    # Instead of immediately updating csets_lst, collect deferred updates
                    for cnodes, cname in csets_old.items():
                        if faceSet.issubset(cnodes):
                            # Defer the update for this face: note the combined name from csets_old
                            newBCFaces.append((faceSet, cname[0], faceVal))

                    # Add the new face to the appropriate element list and bump the face count
                    elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))
                    nFaces[faceVal] += 1

            # After processing all sub-elements, merge the deferred BC face updates
            for newFace, faceName, faceVal in newBCFaces:
                for subFace, faceIndex in subFaces:
                    if subFace == newFace:
                        csets_lst.setdefault(faceName, [[], []])
                        csets_lst[faceName][faceVal].append(faceIndex)

            # Append the split tetrahedral sub-elements.
            elems_lst.setdefault(ElemType, []).extend(subElems)

            # Update the progress bar after processing this element
            bar.step()

    # Close the progress bar
    bar.close()

    # Convert lists to NumPy arrays for elems_new and csets_new
    elems_new = {}
    csets_new = {}

    for key in elems_lst:
        if   isinstance(elems_lst[key], list) and     elems_lst[key]:  # noqa: E271
            # Convert the list of accumulated arrays/lists into a single NumPy array
            elems_new[key] = np.array(elems_lst[key], dtype=int)
        elif isinstance(elems_lst[key], list) and not elems_lst[key]:
            # Determine the expected number of columns
            elems_new[key] = np.empty((0, faceNum[faceType.index(key)]), dtype=int)

    for key in csets_lst:
        csets_new[key] = [np.array(lst, dtype=int) for lst in csets_lst[key]]

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251

    hopout.sep()

    return mesh


@cache
def hexa_faces(order: int) -> list[np.ndarray]:
    """ Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
        return a list of new hexahedral face connectivity lists.
    """
    match order:
        case 1:
            return [np.array([  0,  1,  2,  3], dtype=int),
                    np.array([  0,  1,  5,  4], dtype=int),
                    np.array([  1,  2,  6,  5], dtype=int),
                    np.array([  2,  6,  7,  3], dtype=int),
                    np.array([  0,  4,  7,  3], dtype=int),
                    np.array([  4,  5,  6,  7], dtype=int)]
        case 2:
            return [np.array([  0,  1,  2,  3,  8,  9, 10, 11, 24], dtype=int),
                    np.array([  0,  1,  5,  4,  8, 17, 12, 16, 22], dtype=int),
                    np.array([  1,  2,  6,  5,  9, 18, 13, 17, 21], dtype=int),
                    np.array([  2,  6,  7,  3, 18, 14, 19, 10, 23], dtype=int),
                    np.array([  0,  4,  7,  3, 16, 15, 19, 11, 20], dtype=int),
                    np.array([  4,  5,  6,  7, 12, 13, 14, 15, 25], dtype=int)]
        case 3:
            return [np.array([  0,  1,  2,  3, *range( 8, 10), *range(10, 12),          *range(12, 14),  *reversed(range(14, 16)), 48, *reversed(range(50, 52)), 49], dtype=int),  # noqa: E501
                    np.array([  0,  1,  5,  4, *range( 8, 10), *range(26, 28), *reversed(range(16, 18)), *reversed(range(24, 26)), 40,          *range(41, 43) , 43], dtype=int),  # noqa: E501
                    np.array([  1,  2,  6,  5, *range(10, 12), *range(28, 30), *reversed(range(18, 20)), *reversed(range(26, 28)), 36,          *range(37, 39) , 39], dtype=int),  # noqa: E501
                    np.array([  2,  6,  7,  3, *range(28, 30), *range(20, 22), *reversed(range(30, 32)), *reversed(range(12, 14)), 44, *reversed(range(46, 48)), 45], dtype=int),  # noqa: E501
                    np.array([  0,  4,  7,  3, *range(24, 26), *range(22, 24), *reversed(range(32, 34)), *reversed(range(14, 16)), 32,          *range(33, 35) , 35], dtype=int),  # noqa: E501
                    np.array([  4,  5,  6,  7, *range(16, 18), *range(18, 20),          *range(20, 22) , *reversed(range(22, 24)), 52,          *range(53, 55) , 54], dtype=int)]  # noqa: E501
        case 4:
            return [np.array([  0,  1,  2,  3, *range( 8, 11), *range(11, 14),          *range(14, 17) , *reversed(range(17, 20)), 80, *reversed(range(81, 84)), 87, *reversed(range(84, 87)), 88], dtype=int),  # noqa: E501
                    np.array([  0,  1,  5,  4, *range( 8, 11), *range(35, 38), *reversed(range(20, 23)), *reversed(range(32, 35)), 62,          *range(63, 66) , 66,          *range(67, 70) , 70], dtype=int),  # noqa: E501
                    np.array([  1,  2,  6,  5, *range(11, 14), *range(38, 41), *reversed(range(23, 26)), *reversed(range(35, 38)), 53,          *range(54, 57) , 57,          *range(58, 61) , 61], dtype=int),  # noqa: E501
                    np.array([  2,  6,  7,  3, *range(38, 41), *range(26, 29), *reversed(range(41, 44)), *reversed(range(14, 17)), 71, *reversed(range(72, 75)), 78, *reversed(range(75, 78)), 79], dtype=int),  # noqa: E501
                    np.array([  0,  4,  7,  3, *range(32, 35), *range(29, 32), *reversed(range(41, 44)), *reversed(range(17, 20)), 44,          *range(45, 48) , 48,          *range(49, 52) , 52], dtype=int),  # noqa: E501
                    np.array([  4,  5,  6,  7, *range(20, 23), *range(23, 26),          *range(26, 29) , *reversed(range(29, 32)), 89,          *range(90, 93) , 93,          *range(94, 97) , 97], dtype=int)]  # noqa: E501
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def tetra_faces(order: int) -> list[np.ndarray]:
    """
    Given the tetrahedral indices, return the 4 triangular faces as tuples
    """
    match order:
        case 1:
            return [np.array([  0,  1,  2], dtype=int),
                    np.array([  0,  1,  3], dtype=int),
                    np.array([  0,  2,  3], dtype=int),
                    np.array([  1,  2,  3], dtype=int)]
        case 2:
            return [np.array([  0,  1,  2,  4,  5,  6], dtype=int),
                    np.array([  0,  1,  3,  4,  8,  7], dtype=int),
                    np.array([  0,  2,  3,  6,  9,  7], dtype=int),
                    np.array([  1,  2,  3,  5,  9,  8], dtype=int)]
        case 4:
            return [np.array([  0,  1,  2,  *range( 4, 13)          , *range(31, 34)], dtype=int),
                    np.array([  0,  1,  3,  *range( 4,  7)          , *range(16, 19), *reversed(range(13, 16)), *range(22, 25)], dtype=int),  # noqa: E501
                    np.array([  0,  2,  3,  *reversed(range(10, 13)), *range(19, 22), *reversed(range(13, 16)), *range(28, 31)], dtype=int),  # noqa: E501
                    np.array([  1,  2,  3,  *range( 7, 10)          , *range(19, 22), *reversed(range(16, 19)), *range(25, 28)], dtype=int)]  # noqa: E501
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def pyram_faces(order: int) -> list[np.ndarray]:
    """
    Given the pyramid corner indices, return the 4 triangular faces and 1 quadrilateral face as tuples
    """
    match order:
        case 1:
            return [# Triangular faces  # noqa: E261
                    np.array([  0,  1,  4], dtype=int),
                    np.array([  1,  2,  4], dtype=int),
                    np.array([  2,  3,  4], dtype=int),
                    np.array([  3,  0,  4], dtype=int),
                    # Quadrilateral face
                    np.array([  0,  1,  2,  3], dtype=int)]
        case 2:
            return [# Triangular faces  # noqa: E261
                    np.array([  0,  1,  4,  5, 10,  9], dtype=int),  # 8, 22,16
                    np.array([  1,  2,  4,  6, 11, 10], dtype=int),  # 9, 26,22
                    np.array([  2,  3,  4,  7, 12, 11], dtype=int),  # 10,20,26
                    np.array([  3,  0,  4,  8,  9, 12], dtype=int),  # 11,16,20
                    # Quadrilateral face
                    np.array([  0,  1,  2,  3,  5,  6,  7,  8, 13], dtype=int)]
        case 4:
            return [# Triangular faces  # noqa: E261
                    np.array([  0,  1,  4,  *range( 4,  7), *range(19, 22), *reversed(range(16, 19)), *range(28, 31)], dtype=int),
                    np.array([  1,  2,  4,  *range( 7, 10), *range(22, 25), *reversed(range(19, 22)), *range(31, 34)], dtype=int),
                    np.array([  2,  3,  4,  *range(10, 13), *range(25, 28), *reversed(range(22, 25)), *range(34, 37)], dtype=int),
                    np.array([  3,  0,  4,  *range(13, 16), *range(16, 19), *reversed(range(25, 28)), *range(37, 40)], dtype=int),
                    # Quadrilateral face
                    np.array([ 0,  1,  2,  3, *range(5, 17), *range(41, 50)], dtype=int)]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def prism_faces(order: int) -> list[np.ndarray]:
    """
    Given the 6 prism corner indices, return the 2 triangular and 3 quadrilateral faces as tuples.
    """
    match order:
        case 1:
            return [# Triangular faces  # noqa: E261
                    np.array([  0,  1,  2], dtype=int),
                    np.array([  3,  4,  5], dtype=int),
                    # Quadrilateral faces
                    np.array([  0,  1,  4,  3], dtype=int),
                    np.array([  1,  2,  5,  4], dtype=int),
                    np.array([  2,  0,  3,  5], dtype=int)]
        case 2:
            return [# Triangular faces  # noqa: E261
                    np.array([  0,  1,  2,  6,  7,  8], dtype=int),
                    np.array([  3,  4,  5,  9, 10, 11], dtype=int),
                    # Quadrilateral faces
                    np.array([  0,  1,  4,  3,  6, 13,  9, 12, 15], dtype=int),
                    np.array([  1,  2,  5,  4,  7, 14, 10, 13, 16], dtype=int),
                    np.array([  2,  0,  3,  5,  8, 12, 11, 14, 17], dtype=int)]
        case 4:
            return [# Triangular faces  # noqa: E261
                    np.array([  0, 1, 2, *range( 6, 15), *range(63, 66)], dtype=int),  # z-
                    np.array([  3, 4, 5, *range(15, 24), *range(60, 63)], dtype=int),  # z+
                    # Quadrilateral faces
                    np.array([  0, 1, 4, 3, *range( 6,  9), *range(27, 30), *reversed(range(15, 18)), *reversed(range(24, 27)), *range(33, 42)], dtype=int),  # noqa: E501
                    np.array([  1, 2, 5, 4, *range( 9, 12), *range(30, 33), *reversed(range(18, 21)), *reversed(range(27, 30)), *range(42, 51)], dtype=int),  # noqa: E501
                    np.array([  2, 0, 3, 5, *range(12, 15), *range(24, 27), *reversed(range(21, 24)), *reversed(range(30, 33)), *range(51, 60)], dtype=int)]  # noqa: E501
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def pyram_to_tet_faces(order: int) -> list[np.ndarray]:
    """ Given the 4 corner node indices of a single tetrahedral element (indexed 0..3),
        return the 4 triangular faces and the 12 quadrilateral faces.
    """
    match order:
        case 1:
            newFaces = [np.array([  0,  1,  3], dtype=int),
                        np.array([  1,  2,  3], dtype=int),
                        np.array([  2,  0,  3], dtype=int),
                        np.array([  0,  1,  2], dtype=int)]
        case 2:
            newFaces = [np.array([  0,  1,  3,  4,  8,  7], dtype=int),
                        np.array([  1,  2,  3,  5,  9,  8], dtype=int),
                        np.array([  2,  0,  3,  6,  7,  9], dtype=int),
                        np.array([  0,  1,  2,  4,  5,  6], dtype=int)]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

    return newFaces


@cache
def pyram_to_tet_split(order: int) -> list[tuple]:
    """ Given the 4 corner node indices of a single pyramid element (indexed 0..3),
        return a list of new tetrahedron element connectivity lists.
    """
    match order:
        case 1:
            return [(0,  1,  3,  4),
                    (2,  3,  1,  4),
                   ]
        case 2:
            return [(0,  1,  3,  4,  5, 13,  8,  9, 10, 12),
                    (2,  3,  1,  4,  7, 13,  6, 11, 12, 10),
                   ]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def NDOFperElemType(elemType: str, nGeo: int) -> int:
    """ Calculate the number of degrees of freedom for a given element type
    """
    match elemType:
        case _ if elemType.startswith('triangle'):
            return round((nGeo+1)*(nGeo+2)/2.)
        case _ if elemType.startswith('quad'):
            return round((nGeo+1)**2)
        case _ if elemType.startswith('tetra'):
            return round((nGeo+1)*(nGeo+2)*(nGeo+3)/6.)
        case _ if elemType.startswith('pyramid'):
            return round((nGeo+1)*(nGeo+2)*(2*nGeo+3)/6.)
        case _ if elemType.startswith('wedge'):
            return round((nGeo+1)**2 *(nGeo+2)/2.)
        case _ if elemType.startswith('hexahedron'):
            return round((nGeo+1)**3)
        case _:
            raise ValueError(f'Unknown element type {elemType}')
