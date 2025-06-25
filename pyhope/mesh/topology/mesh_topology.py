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
from collections import defaultdict
from functools import cache
from typing import cast
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


def MeshChangeElemType(mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_progress import ProgressBar
    from pyhope.mesh.mesh_vars import ELEMTYPE, nGeo
    # ------------------------------------------------------

    # Split hexahedral elements if requested
    nZones    = mesh_vars.nZones
    elemTypes = mesh_vars.elemTypes
    elemNames = ['' for _ in range(nZones)]  # noqa: E271

    # No element types given
    if len(elemTypes) == 0:
        return mesh

    # Fully hexahedral mesh
    if all(elemType % 100 == 8 for elemType in elemTypes):
        return mesh

    # Simplex elements requested
    if any(elemType % 100 != 8 for elemType in elemTypes):
        if mesh_vars.nGeo > 4:
            hopout.error('Non-hexahedral elements are not supported for nGeo > 4, exiting...')

    hopout.info('Converting hexahedral elements to simplex elements')

    # Instantiate ELEMTYPE
    elemTypeInam = ELEMTYPE().inam

    for i in range(nZones):
        if nGeo == 1:
            elemNames[i] = elemTypeInam[elemTypes[i]][0]
        else:
            # check whether user entered correct high-order element type
            if elemTypes[i] < 200:
                # Adapt to high-order element type
                elemTypes[i] += 100

            # Get the element name and skip the entries for incomplete 2nd order elements
            try:
                if elemTypes[i] % 100 == 5:    # pyramids (skip 1)
                    elemNames[i] = elemTypeInam[elemTypes[i]][1]
                elif elemTypes[i] % 100 == 6:  # prisms (skip 1)
                    elemNames[i] = elemTypeInam[elemTypes[i]][nGeo-1]
                elif elemTypes[i] % 100 == 8:  # hexahedra (skip 2)
                    elemNames[i] = elemTypeInam[elemTypes[i]][nGeo]
                else:                          # tetrahedra
                    elemNames[i] = elemTypeInam[elemTypes[i]][nGeo-2]
            except IndexError:
                hopout.error('Element type {} not supported for nGeo = {}, exiting...'.format(elemTypes[i], nGeo))

    # Copy original points
    pointl    = cast(list, mesh.points.tolist())
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    # Get base key to distinguish between linear and high-order elements
    ho_key = 100 if nGeo == 1 else 200

    # Set up the element splitting function
    elemSplitter = {ho_key + 4: (split_hex_to_tets , tetra_faces),
                    ho_key + 5: (split_hex_to_pyram, pyram_faces),
                    ho_key + 6: (split_hex_to_prism, prism_faces),
                    # Keep hexahedral elements as they are
                    ho_key + 8: (split_hex_to_hex  , hex_faces  )}

    faceMaper = { ho_key + 4: lambda x: 0,
                  ho_key + 5: lambda x: 0 if x == 0 else 1,
                  ho_key + 6: lambda x: 0 if x == 0 else 1,
                  # Keep hexahedral elements as they are
                  ho_key + 8: lambda x: 1}
    nFace   = (nGeo+1)*(nGeo+2)/2

    # Convert the (quad) boundary cell set into a dictionary
    csets_old = {}

    for cname, cblock in cell_sets.items():
        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID, block in enumerate(cblock):
            if elems_old[blockID].type[:4] != 'quad':
                continue

            # Ignore the volume zones
            if block is None:
                continue

            # Sort them as a set for membership checks
            for face in block:
                nodes = mesh.cells_dict[elems_old[blockID].type][face]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    nPoints  = len(pointl)
    nFaces   = np.zeros(2, dtype=int)
    match nGeo:
        case 1:
            faceType = ['triangle'  , 'quad'  ]
            faceNum  = [          3 ,       4 ]
        case 2:
            faceType = ['triangle6' , 'quad9' ]
            faceNum  = [          6 ,       9 ]
        # case 3:
        #     faceType = ['triangle10', 'quad16']
        #     faceNum  = [         10 ,      16 ]
        case 4:
            faceType = ['triangle15', 'quad25']
            faceNum  = [         15 ,      25 ]
        case _:
            hopout.error('nGeo = {} not supported for element splitting'.format(nGeo))

    # Prepare new cell blocks and new cell_sets
    elems_lst = {ftype: [] for ftype in faceType}
    csets_lst = {}

    # Create the element sets
    meshcells = tuple((k, v) for k, v in mesh.cell_sets_dict.items() if any(key.startswith('hexahedron') for key in v.keys()))

    # If meshcells is empty, we fake it assign it to Zone1
    if len(meshcells) == 0:
        meshcells = tuple(('Zone1', np.array([i for i in range(len(v))])) for k, v in mesh.cells_dict.items()
                                                                                   if k.startswith('hexahedron'))

    nTotalElems = sum(cdata.shape[0] for _, zdata in meshcells for _, cdata in cast(dict, zdata).items())
    bar = ProgressBar(value=nTotalElems, title='│             Processing Elements', length=33, threshold=1000)

    # Build an inverted index to map each node to all face keys (from csets_old) that contain it
    nodeToFace = defaultdict(set)
    for subFace in csets_old:
        for node in subFace:
            nodeToFace[node].add(subFace)

    for iElem, meshcell in enumerate(meshcells):
        _    , mdict = meshcell
        mtype, mcell = list(cast(dict, mdict).keys())[0], list(cast(dict, mdict).values())[0]

        elemType     = elemTypes[iElem]
        elemName     = elemNames[iElem]

        split, faces = elemSplitter.get(elemType, (None, None))
        faceMap      = faceMaper.get(elemType, None)

        # Sanity check
        if faceMap is None:
            raise ValueError('Missing faceMap for element type {}'.format(elemType))

        cdata = mesh.get_cells_type(mtype)[mcell]

        if split is None or faces is None:
            hopout.error('Element type {} not supported for splitting'.format(elemTypes[iElem]), traceback=True)

        elemSplit = split(nGeo)

        # Hex block: Iterate over each element
        for elem in cdata:
            # Pyramids need a center node
            if elemType % 100 == 5:
                # Find the element orientation
                # > The first 8 points (indices) in elem are the CGNS-ordered vertices
                vertices = np.array([pointl[i] for i in elem[:8]])

                # v[0] is origin, v[1] is local x-direction, v[2] is local y-direction, v[3] is local z-direction
                # > This only works for trapezoidal elements
                v        = [vertices[0], vertices[1]-vertices[0], vertices[3]-vertices[0], vertices[4]-vertices[0]]

                # Compute the element center
                center   = np.mean(np.array([pointl[i] for i in elem]), axis=0)

                match nGeo:
                    case 1:
                        # Append the new point to the point list
                        pointl.append(center.tolist())

                        # Overwrite the element with the new indices
                        elem     = np.array(list(elem) + [nPoints])
                        nPoints += 1
                    case 2:
                        # Generate the grid of new points
                        signarr  = [-0.5, 0.5]
                        # Create a 3D grid of sign factors

                        grid = np.array(np.meshgrid(signarr, signarr, signarr)).T.reshape(-1, 3)
                        # Combine the grid with the directional vectors v[1], v[2], v[3]
                        edge_array = center + 0.5 * (grid[:, 0][:, None] * v[1] +
                                                     grid[:, 1][:, None] * v[2] +
                                                     grid[:, 2][:, None] * v[3])
                        # Convert all points to list in one go
                        edges   = edge_array.tolist()
                        # Append the new points to the point list
                        pointl.extend(edges)

                        # Overwrite the element with the new indices
                        elem     = np.array(list(elem) + list(range(nPoints, nPoints+edge_array.shape[0])))
                        nPoints += edge_array.shape[0]
                    case 4:
                        # Generate the grid of new points
                        signarr = [-3./4., -1./4., 1./4., 3./4.]
                        # Create a 3D grid of sign factors
                        grid = np.array(np.meshgrid(signarr, signarr, signarr)).T.reshape(-1, 3)
                        # Combine the grid with the directional vectors v[1], v[2], v[3]
                        edge_array = center + 0.5 * (grid[:, 0][:, None] * v[1] +
                                                     grid[:, 1][:, None] * v[2] +
                                                     grid[:, 2][:, None] * v[3])
                        # Convert all points to list in one go
                        edges   = edge_array.tolist()
                        # Append the new points to the point list
                        pointl.extend(edges)

                        # Overwrite the element with the new indices
                        elem     = np.array(list(elem) + list(range(nPoints, nPoints+edge_array.shape[0])))
                        nPoints += edge_array.shape[0]

            # Split each element into sub-elements
            subElems = elem[elemSplit]

            for subElem in subElems:
                subFaces = tuple(np.array(subElem)[face] for face in faces(nGeo))

                for subFace in subFaces:
                    faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)
                    faceSet = frozenset(subFace)

                    # Get candidate cset keys using the nodes in the face
                    candidate_sets = [nodeToFace[node] for node in faceSet if node in nodeToFace]
                    if not candidate_sets:
                        continue

                    common_candidates = set.intersection(*candidate_sets)
                    for candidate in common_candidates:
                        # Check if the subFace is indeed a subset of the candidate from csets_old
                        if faceSet.issubset(candidate):
                            # Use the associated boundary name
                            # (Assuming all boundary names are stored in a list for this candidate. Adjust if needed.)
                            names = csets_old[candidate]
                            # Update csets_lst for each name in the list.
                            for name in names:
                                csets_lst.setdefault(name, [[], []])
                                csets_lst[name][faceVal].append(nFaces[faceVal])

                    elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))
                    nFaces[faceVal] += 1

            elems_lst.setdefault(elemName, []).extend(subElems)

            # Update the progress bar
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
        csets_new[key] = tuple(np.array(lst, dtype=int) for lst in csets_lst[key])

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251

    hopout.sep()

    return mesh


@cache
def split_hex_to_tets(order: int) -> list[tuple]:
    """
    Given the indices of a single hexahedral element, return a list of new tetra element connectivity lists

    The node numbering convention assumed here (c0, c1, c2, c3, c4, c5, c6, c7) is the usual:
          7-------6
         /|      /|
        4-------5 |
        | 3-----|-2
        |/      |/
        0-------1

    """
    # Perform the 6-tet split of the cube-like cell
    match order:
        case 1:
            # 1. strategy: 6 tets per box, all tets have same volume and angle, not periodic but isotropic
            return [( 0, 2, 3, 4),
                    ( 0, 1, 2, 4),
                    ( 2, 4, 6, 7),
                    ( 2, 4, 5, 6),
                    ( 1, 2, 4, 5),
                    ( 2, 3, 4, 7)]
            # ! 2. strategy: 6 tets per box, split hex into two prisms and each prism into 3 tets, periodic but strongly anisotropic
            # c0, c1, c2, c3, c4, c5, c6, c7 = nodes
            #  return [[0, 1, 3, 4],
            #          [1, 4, 5, 7],
            #          [1, 3, 4, 7],
            #          [1, 2, 5, 7],
            #          [1, 3, 2, 7],
            #          [2, 5, 7, 6]]
        case 2:
            return [( 0,  2,  3,  4,  24,  10,  11,  16,  26,  20),
                    ( 0,  1,  2,  4,   8,   9,  24,  16,  22,  26),
                    ( 2,  4,  6,  7,  26,  25,  18,  23,  15,  14),
                    ( 2,  4,  5,  6,  26,  12,  21,  18,  25,  13),
                    ( 1,  2,  4,  5,   9,  26,  22,  17,  21,  12),
                    ( 2,  3,  4,  7,  10,  20,  26,  23,  19,  15)]
        case 4:
            tetra1 = (  0,   2,   3,   4,  80,  88,  82,  14,  15,  16,  # noqa: E501
                       19,  18,  17,  32,  33,  34, 100, 124, 102,
                       47,  52,  45,  98, 122, 114, 108, 101, 118,
                       44,  51,  48,  84,  85,  81, 109)
            tetra2 = (  0,   1,   2,   4,   8,   9,  10,  11,  12,  13,  # noqa: E501
                       82,  88,  80,  32,  33,  34,  63,  70,  65,
                      100, 124, 102,  62,  66,  69,  99, 107, 120,
                       98, 122, 114,  87,  83,  86, 106)
            tetra3 = (  2,   4,   6,   7, 100, 124, 102,  89,  97,  91,  # noqa: E501
                       40,  39,  38,  71,  79,  73,  29,  30,  31,
                       26,  27,  28, 121, 113, 105,  96,  95,  92,
                       78,  74,  77, 116, 123, 104, 112)
            tetra4 = (  2,   4,   5,   6, 100, 124, 102,  20,  21,  22,  # noqa: E501
                       56,  61,  54,  38,  39,  40,  89,  97,  91,
                       23,  24,  25, 116, 123, 104,  93,  90,  94,
                       58,  59,  55, 119, 110, 103, 111)
            tetra5 = (  1,   2,   4,   5,  11,  12,  13, 100, 124, 102,  # noqa: E501
                       65,  70,  63,  35,  36,  37,  54,  61,  56,
                       20,  21,  22,  53,  57,  60, 119, 110, 103,
                       67,  68,  64,  99, 107, 120, 115)
            tetra6 = (  2,   3,   4,   7,  14,  15,  16,  47,  52,  45,  # noqa: E501
                      102, 124, 100,  71,  79,  73,  41,  42,  43,
                       29,  30,  31,  75,  72,  76,  50,  49,  46,
                      121, 113, 105, 108, 101, 118, 117)

            return [tetra1, tetra2, tetra3, tetra4, tetra5, tetra6]
        case _:
            # Lazy-load local import
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)


@cache
def tetra_faces(order: int) -> tuple[np.ndarray, ...]:
    """
    Given the tetrahedral indices, return the 4 triangular faces as tuples
    """
    match order:
        case 1:
            return (np.array((  0,  1,  2), dtype=int),
                    np.array((  0,  1,  3), dtype=int),
                    np.array((  0,  2,  3), dtype=int),
                    np.array((  1,  2,  3), dtype=int))
        case 2:
            return (np.array((  0,  1,  2,  4,  5,  6), dtype=int),
                    np.array((  0,  1,  3,  4,  8,  7), dtype=int),
                    np.array((  0,  2,  3,  6,  9,  7), dtype=int),
                    np.array((  1,  2,  3,  5,  9,  8), dtype=int))
        case 4:
            return (np.array((  0,  1,  2,  *range( 4, 13)          , *range(31, 34)), dtype=int),
                    np.array((  0,  1,  3,  *range( 4,  7)          , *range(16, 19), *reversed(range(13, 16)), *range(22, 25)), dtype=int),  # noqa: E501
                    np.array((  0,  2,  3,  *reversed(range(10, 13)), *range(19, 22), *reversed(range(13, 16)), *range(28, 31)), dtype=int),  # noqa: E501
                    np.array((  1,  2,  3,  *range( 7, 10)          , *range(19, 22), *reversed(range(16, 19)), *range(25, 28)), dtype=int))  # noqa: E501
        case _:
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)


@cache
def split_hex_to_pyram(order: int) -> list[tuple[int, ...]]:
    """
    Given the indices of a single hexahedral element, return a list of new pyramid element connectivity lists
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    match order:
        case 1:
            # Perform the 6-pyramid split of the cube-like cell
            return [( 0,  1,  2,  3,  8),
                    ( 0,  4,  5,  1,  8),
                    ( 1,  5,  6,  2,  8),
                    ( 0,  3,  7,  4,  8),
                    ( 4,  7,  6,  5,  8),
                    ( 6,  7,  3,  2,  8)]
            # 3-pyramid split
            # return [( 0, 1, 2, 3, 4),
            #         ( 1, 5, 6, 2, 4),
            #         ( 6, 7, 3, 2, 4)]
        case 2:
            # Perform the 6-pyramid split of the cube-like cell
            return [(  0,  1,  2,  3, 26,  8,  9, 10, 11, 27, 28, 30, 29, 24),
                    (  0,  4,  5,  1, 26, 16, 12, 17,  8, 27, 31, 32, 28, 22),
                    (  1,  5,  6,  2, 26, 17, 13, 18,  9, 28, 32, 34, 30, 21),
                    (  0,  3,  7,  4, 26, 11, 19, 15, 16, 27, 29, 33, 31, 20),
                    (  4,  7,  6,  5, 26, 15, 14, 13, 12, 31, 33, 34, 32, 25),
                    (  6,  7,  3,  2, 26, 14, 19, 10, 18, 34, 33, 29, 30, 23)]
            # 3-pyramid split
            # return [(  0,  1,  2,  3,  4,  8,  9, 10, 11, 16, 22, 26, 20, 24),
            #         (  1,  5,  6,  2,  4, 17, 13, 18,  9, 22, 12, 25, 26, 21),
            #         (  6,  7,  3,  2,  4, 14, 19, 10, 18, 25, 15, 20, 26, 23)]
        case 4:
            return [(  0,   1,   2,   3, 124, *range( 8,  17), *reversed(range(17, 20)),
                     125,  98, 146, 128,  99, 147, 140, 100, 151, 137, 101, 150, 126,
                     127, 106, 132, 136, 107, 138, 139, 108, 129, 133, 109,  80,  83,
                      82,  81,  87,  86,  85,  84,  88, 130, 131, 135, 134, 122),
                    (  0,   4,   5,   1, 124, *range(32,  35),          *range(20,  23),
                      37,  36,  35,  10,   9,   8, 125,  98, 146, 173, 102, 162, 176,
                     103, 163, 128,  99, 147, 141, 157, 114, 174, 175, 110, 144, 160,
                     115, 126, 127, 106,  62,  65,  64,  63,  69,  68,  67,  66,  70, 142, 158, 159, 143, 120),
                    (  1,   5,   6,   2, 124, *range(35,  38),          *range(23,  26),
                      40,  39,  38,  13,  12,  11, 128,  99, 147, 176, 103, 163, 188,
                     104, 167, 140, 100, 151, 144, 160, 115, 180, 184, 111, 156, 172,
                     116, 132, 136, 107,  53,  56,  55,  54,  60,  59,  58,  57,  61, 148, 164, 168, 152, 119),
                    (  0,   3,   7,   4, 124,  17,  18,  19, *range(41,  44), *reversed(range(29, 32)),
                      34,  33,  32, 125,  98, 146, 137, 101, 150, 185, 105, 166, 173,
                     102, 162, 129, 133, 109, 153, 169, 117, 177, 181, 113, 141, 157,
                     114,  44,  47,  46,  45,  51,  50,  49,  48,  52, 145, 149, 165, 161, 118),
                    (  4,   7,   6,   5, 124, *range(29,  32), *reversed(range(26, 29)),
                      25,  24,  23,  22,  21,  20, 173, 102, 162, 185, 105, 166, 188,
                     104, 167, 176, 103, 163, 177, 181, 113, 186, 187, 112, 180, 184,
                     111, 174, 175, 110,  89,  92,  91,  90,  96,  95,  94,  93,  97, 178, 182, 183, 179, 123),
                    (  6,   7,   3,   2, 124, *range(26,  29), *reversed(range(41, 44)),
                      16,  15,  14,  38,  39,  40, 188, 104, 167, 185, 105, 166, 137,
                     101, 150, 140, 100, 151, 187, 186, 112, 169, 153, 117, 139, 138,
                     108, 172, 156, 116,  74,  73,  72,  71,  77,  76,  75,  78,  79, 171, 170, 154, 155, 121)]
        case _:
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)


@cache
def pyram_faces(order: int) -> tuple[np.ndarray, ...]:
    """
    Given the pyramid corner indices, return the 4 triangular faces and 1 quadrilateral face as tuples
    """
    match order:
        case 1:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  4), dtype=int),
                    np.array((  1,  2,  4), dtype=int),
                    np.array((  2,  3,  4), dtype=int),
                    np.array((  3,  0,  4), dtype=int),
                    # Quadrilateral face
                    np.array((  0,  1,  2,  3), dtype=int))
        case 2:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  4,  5, 10,  9), dtype=int),  # 8, 22,16
                    np.array((  1,  2,  4,  6, 11, 10), dtype=int),  # 9, 26,22
                    np.array((  2,  3,  4,  7, 12, 11), dtype=int),  # 10,20,26
                    np.array((  3,  0,  4,  8,  9, 12), dtype=int),  # 11,16,20
                    # Quadrilateral face
                    np.array((  0,  1,  2,  3,  5,  6,  7,  8, 13), dtype=int))
        case 4:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  4,  *range( 4,  7), *range(19, 22), *reversed(range(16, 19)), *range(28, 31)), dtype=int),  # noqa: E501
                    np.array((  1,  2,  4,  *range( 7, 10), *range(22, 25), *reversed(range(19, 22)), *range(31, 34)), dtype=int),  # noqa: E501
                    np.array((  2,  3,  4,  *range(10, 13), *range(25, 28), *reversed(range(22, 25)), *range(34, 37)), dtype=int),  # noqa: E501
                    np.array((  3,  0,  4,  *range(13, 16), *range(16, 19), *reversed(range(25, 28)), *range(37, 40)), dtype=int),  # noqa: E501
                    # Quadrilateral face
                    np.array(( 0,  1,  2,  3, *range(5, 17), *range(41, 50)), dtype=int))
        case _:
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)


@cache
def split_hex_to_prism(order: int) -> list[tuple[int, ...]]:
    """
    Given the indices of a single hexahedral element, return a list of new prism element connectivity lists
    """
    match order:
        case 1:
            #  return [( 0,  1,  3,  4,  5,  7),
            #          ( 1,  2,  3,  5,  6,  7)]
            return [( 0,  1,  2,  4,  5,  6),
                    ( 0,  2,  3,  4,  6,  7)]
        case 2:
            #  HEXA: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 24 22 21 23 20 25 26]
            #  return [(  0,  1,  3,  4,  5,  7,  8, 24, 11, 12, 25, 15, 16, 17, 19, 22, 26, 20),
            #          (  1,  2,  3,  5,  6,  7,  9, 10, 24, 13, 14, 25, 17, 18, 19, 21, 23, 26)]
            return [(  0,  1,  2,  4,  5,  6,  8,  9, 24, 12, 13, 25, 16, 17, 18, 22, 21, 26),
                    (  0,  2,  3,  4,  6,  7, 24, 10, 11, 25, 14, 15, 16, 18, 19, 26, 23, 20)]
        case 4:
            # prism1 = (   0,   1,   3,   4,   5,   7,
            #              8,   9,  10,  83,  88,  81,  19,  18,  17,            # 6 vertices
            #             20,  21,  22,  90,  97,  92,  31,  30,  29,            # Edge
            #             32,  33,  34,  35,  36,  37,  41,  42,  43,            # Edge
            #             62,  63,  64,  65,  66,  67,  68,  69,  70,            # Face 1
            #             99, 101, 105, 103, 122, 117, 123, 115, 124,            # Face 2
            #             47,  44,  45,  46,  51,  48,  49,  50,  52,            # Face 3
            #             89,  93,  96,                                          # Face 4
            #             80,  87,  84,                                          # Face 5
            #             98, 106, 109, 114, 120, 118, 102, 110, 113)            # Volume
            prism1 = (   0,   1,   3,   4,   5,   7,                           # 6 vertices
                        *range( 8, 11), 83, 88, 81, *reversed(range(17, 20)),  # Edges
                        *range(20, 23), 90, 97, 92, *reversed(range(29, 32)),  # Edges
                        *range(32, 35), 35, 36, 37, *reversed(range(41, 44)),  # Face 1
                        *range(62, 65), 65, 66, 67,          *range(68, 71) ,  # Face 2
                         99,  101, 105, 103, 122,  117, 123, 115, 124,         # Face 3
                         47, *range(44, 47),  51, *range(48, 51),  52,         # Face 4
                         89,  93,  96, 80,  87,  84,                           # Face 5
                         98, 106, 109, 114, 120, 118, 102, 110, 113)           # Volume
            # prism2 = (   1,   2,   3,   5,   6,   7,                           # 6 vertices
            #             11,  12,  13,  14,  15,  16,  81,  88,  83,            # Edges
            #             23,  24,  25,  26,  27,  28,  92,  97,  90,            # Edges
            #             35,  36,  37,  38,  39,  40,  41,  42,  43,            # Face 1
            #             53,  54,  55,  56,  57,  58,  59,  60,  61,            # Face 2
            #             71,  72,  73,  74,  75,  76,  77,  78,  79,            # Face 3
            #            101,  99, 103, 105, 122, 115, 123, 117, 124,            # Face 4
            #             94,  91,  95, 86,  82,  85,                            # Face 5
            #            107, 100, 108, 119, 116, 121, 111, 104, 112)            # Volume
            prism2 = (   1,   2,   3,   5,   6,   7,                           # 6 vertices
                        *range(11, 14), *range(14, 17), 81, 88, 83,            # Edges
                        *range(23, 26), *range(26, 29), 92, 97, 90,            # Edges
                        *range(35, 38), *range(38, 41), 41, 42, 43,            # Face 1
                        *range(53, 56), *range(56, 59), 59, 60, 61,            # Face 2
                        *range(71, 74), *range(74, 77), 77, 78, 79,            # Face 3
                        101,  99, 103, 105, 122, 115, 123, 117, 124,           # Face 4
                         94,  91,  95,  86,  82,  85,                          # Face 5
                        107, 100, 108, 119, 116, 121, 111, 104, 112)           # Volume
            return [prism1, prism2]

        case _:
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)


@cache
def prism_faces(order: int) -> tuple[np.ndarray, ...]:
    """
    Given the 6 prism corner indices, return the 2 triangular and 3 quadrilateral faces as tuples.
    """
    match order:
        case 1:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  2    ), dtype=int),
                    np.array((  3,  4,  5    ), dtype=int),
                    # Quadrilateral faces
                    np.array((  0,  1,  4,  3), dtype=int),
                    np.array((  1,  2,  5,  4), dtype=int),
                    np.array((  2,  0,  3,  5), dtype=int))
        case 2:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  2,  6,  7,  8            ), dtype=int),
                    np.array((  3,  4,  5,  9, 10, 11            ), dtype=int),
                    # Quadrilateral faces
                    np.array((  0,  1,  4,  3,  6, 13,  9, 12, 15), dtype=int),
                    np.array((  1,  2,  5,  4,  7, 14, 10, 13, 16), dtype=int),
                    np.array((  2,  0,  3,  5,  8, 12, 11, 14, 17), dtype=int))
        case 4:
            return (# Triangular faces  # noqa: E261
                    np.array((  0, 1, 2, *range( 6, 15), *range(63, 66)), dtype=int),  # z-
                    np.array((  3, 4, 5, *range(15, 24), *range(60, 63)), dtype=int),  # z+
                    # Quadrilateral faces
                    np.array((  0, 1, 4, 3, *range( 6,  9), *range(27, 30), *reversed(range(15, 18)), *reversed(range(24, 27)), *range(33, 42)), dtype=int),  # noqa: E501
                    np.array((  1, 2, 5, 4, *range( 9, 12), *range(30, 33), *reversed(range(18, 21)), *reversed(range(27, 30)), *range(42, 51)), dtype=int),  # noqa: E501
                    np.array((  2, 0, 3, 5, *range(12, 15), *range(24, 27), *reversed(range(21, 24)), *reversed(range(30, 33)), *range(51, 60)), dtype=int))  # noqa: E501
        case _:
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)


# Dummy function for hexahedral elements
@cache
def split_hex_to_hex(order: int) -> list[tuple[int, ...]]:
    nodes = np.arange((order + 1) ** 3)
    return [tuple(nodes)]


# Dummy function for hexahedral elements
@cache
def hex_faces(order: int) -> tuple[np.ndarray, ...]:
    """ Given the indices of a hexahedral element, return the 6 faces as tuples
    """
    match order:
        case 1:
            return (np.array((  0,  1,  2,  3), dtype=int),
                    np.array((  0,  1,  5,  4), dtype=int),
                    np.array((  1,  2,  6,  5), dtype=int),
                    np.array((  2,  6,  7,  3), dtype=int),
                    np.array((  0,  4,  7,  3), dtype=int),
                    np.array((  4,  5,  6,  7), dtype=int))
        case 2:
            return (np.array((  0,  1,  2,  3,  8,  9, 10, 11, 24), dtype=int),
                    np.array((  0,  1,  5,  4,  8, 17, 12, 16, 22), dtype=int),
                    np.array((  1,  2,  6,  5,  9, 18, 13, 17, 21), dtype=int),
                    np.array((  2,  6,  7,  3, 18, 14, 19, 10, 23), dtype=int),
                    np.array((  0,  4,  7,  3, 16, 15, 19, 11, 20), dtype=int),
                    np.array((  4,  5,  6,  7, 12, 13, 14, 15, 25), dtype=int))
        case 3:
            return (np.array((  0,  1,  2,  3, *range( 8, 10), *range(10, 12),          *range(12, 14),  *reversed(range(14, 16)), 48, *reversed(range(50, 52)), 49), dtype=int),  # noqa: E501
                    np.array((  0,  1,  5,  4, *range( 8, 10), *range(26, 28), *reversed(range(16, 18)), *reversed(range(24, 26)), 40,          *range(41, 43) , 43), dtype=int),  # noqa: E501
                    np.array((  1,  2,  6,  5, *range(10, 12), *range(28, 30), *reversed(range(18, 20)), *reversed(range(26, 28)), 36,          *range(37, 39) , 39), dtype=int),  # noqa: E501
                    np.array((  2,  6,  7,  3, *range(28, 30), *range(20, 22), *reversed(range(30, 32)), *reversed(range(12, 14)), 44, *reversed(range(46, 48)), 45), dtype=int),  # noqa: E501
                    np.array((  0,  4,  7,  3, *range(24, 26), *range(22, 24), *reversed(range(32, 34)), *reversed(range(14, 16)), 32,          *range(33, 35) , 35), dtype=int),  # noqa: E501
                    np.array((  4,  5,  6,  7, *range(16, 18), *range(18, 20),          *range(20, 22) , *reversed(range(22, 24)), 52,          *range(53, 55) , 54), dtype=int))  # noqa: E501
        case 4:
            return (np.array((  0,  1,  2,  3, *range( 8, 11), *range(11, 14),          *range(14, 17) , *reversed(range(17, 20)), 80, *reversed(range(81, 84)), 87, *reversed(range(84, 87)), 88), dtype=int),  # noqa: E501
                    np.array((  0,  1,  5,  4, *range( 8, 11), *range(35, 38), *reversed(range(20, 23)), *reversed(range(32, 35)), 62,          *range(63, 66) , 66,          *range(67, 70) , 70), dtype=int),  # noqa: E501
                    np.array((  1,  2,  6,  5, *range(11, 14), *range(38, 41), *reversed(range(23, 26)), *reversed(range(35, 38)), 53,          *range(54, 57) , 57,          *range(58, 61) , 61), dtype=int),  # noqa: E501
                    np.array((  2,  6,  7,  3, *range(38, 41), *range(26, 29), *reversed(range(41, 44)), *reversed(range(14, 17)), 71, *reversed(range(72, 75)), 78, *reversed(range(75, 78)), 79), dtype=int),  # noqa: E501
                    np.array((  0,  4,  7,  3, *range(32, 35), *range(29, 32), *reversed(range(41, 44)), *reversed(range(17, 20)), 44,          *range(45, 48) , 48,          *range(49, 52) , 52), dtype=int),  # noqa: E501
                    np.array((  4,  5,  6,  7, *range(20, 23), *range(23, 26),          *range(26, 29) , *reversed(range(29, 32)), 89,          *range(90, 93) , 93,          *range(94, 97) , 97), dtype=int))  # noqa: E501
        case _:
            import pyhope.output.output as hopout
            hopout.error('Order {} not supported for element splitting'.format(order), traceback=True)
