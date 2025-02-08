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
    from pyhope.mesh.mesh_vars import ELEMTYPE, nGeo
    from pyhope.readintools.readintools import GetIntFromStr
    # ------------------------------------------------------

    # Split hexahedral elements if requested
    elemType = GetIntFromStr('ElemType')

    match elemType % 100:
        case 8:
            return mesh
        case _:
            # Simplex elements requested
            if mesh_vars.nGeo > 4:
                hopout.warning('Non-hexahedral elements are not supported for nGeo > 4, exiting...')
                sys.exit(1)

    # Copy original points
    points    = mesh.points.copy()
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    # Get base key to distinguish between linear and high-order elements
    ho_key = 100 if nGeo == 1 else 200

    # Prepare new cell blocks and new cell_sets
    elems_new = {}
    csets_new = {}
    if nGeo == 1:
        elemName  = ELEMTYPE.inam[elemType][0]
    else:
        # check whether user entered correct high-order element type
        if elemType < 200:
            # Adapt to high-order element type
            elemType += 100

        # Get the element name and skip the entries for incomplete 2nd order elements
        try:
            if elemType % 100 == 5:    # pyramids (skip 1)
                elemName = ELEMTYPE.inam[elemType][nGeo-1]
            elif elemType % 100 == 6:  # prisms (skip 1)
                elemName = ELEMTYPE.inam[elemType][nGeo-1]
            elif elemType % 100 == 8:  # hexahedra (skip 2)
                elemName = ELEMTYPE.inam[elemType][nGeo]
            else:                      # tetrahedra
                elemName = ELEMTYPE.inam[elemType][nGeo-2]
        except IndexError:
            hopout.warning('Element type {} not supported for nGeo = {}, exiting...'.format(elemType, nGeo))
            sys.exit(1)

    # Convert the (quad) boundary cell set into a dictionary
    csets_old = {}

    # Calculate the offset for the quad cells
    offset    = 0
    for elems in elems_old:
        if any(sub in elems.type for sub in {'vertex', 'line'}):
            offset += len(elems.data)

    for cname, cblock in cell_sets.items():
        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID, block in enumerate(cblock):
            if elems_old[blockID].type[:4] != 'quad':
                continue

            # Sort them as a set for membership checks
            for face in block:
                nodes = mesh.cells_dict[elems_old[blockID].type][face - offset]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    # Set up the element splitting function
    elemSplitter = {ho_key + 4: (split_hex_to_tets , tetra_faces),
                    ho_key + 5: (split_hex_to_pyram, pyram_faces),
                    ho_key + 6: (split_hex_to_prism, prism_faces)}
    split, faces = elemSplitter.get(elemType, (None, None))

    if split is None or faces is None:
        hopout.warning('Element type {} not supported for splitting'.format(elemType))
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    nPoints  = len(points)
    nFaces   = np.zeros(2)
    match nGeo:
        case 1:
            faceType = ['triangle' , 'quad']
        case 2:
            faceType = ['triangle6', 'quad9']
        case 4:
            faceType = ['triangle15', 'quad25']
        case _:
            hopout.warning('nGeo = {} not supported for element splitting'.format(nGeo))
            sys.exit(1)

    faceMaper = { ho_key + 4: lambda x: 0,
                  ho_key + 5: lambda x: 0 if x == 0 else 1,
                  ho_key + 6: lambda x: 0 if x == 0 else 1}
    faceMap   = faceMaper.get(elemType, None)

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        if ctype[:10] != 'hexahedron':
            # Fill the cube recursively from the outside to the inside
            continue

        # Hex block: Split each element
        for elem in cdata:
            # Pyramids need a center node
            if elemType % 100 == 5:
                match nGeo:
                    case 1:
                        center   = np.mean(  points[elem]  , axis=0)
                        center   = np.expand_dims(center   , axis=0)
                        points   = np.append(points, center, axis=0)
                        subElems = split(elem, nPoints, [], nGeo)
                        nPoints += 1
                    case 2:
                        edges = []
                        center   = np.mean(points[elem], axis=0)
                        minext   = np.min( points[elem], axis=0)
                        edges    = np.zeros((8, 3))
                        signarr  = [-0.5, 0.5]
                        count = 0
                        for k in signarr:
                            for j in signarr:
                                for i in signarr:
                                    edges[count, :] = [center[0]+i*abs(center[0]-minext[0]),
                                                       center[1]+j*abs(center[1]-minext[1]),
                                                       center[2]+k*abs(center[2]-minext[2])]
                                    count+=1
                        points   = np.append(points, edges, axis=0)
                        subElems = split(elem, nPoints, np.arange(nPoints, nPoints+8), nGeo)
                        nPoints += 8
                    # case _:
                    #     hopout.warning('nGeo = {} not supported for element splitting'.format(nGeo))
                    #     traceback.print_stack(file=sys.stdout)
                    #     sys.exit(1)
            else:
                subElems = split(elem, nGeo)

            for subElem in subElems:
                subFaces = [np.array(subElem)[face] for face in faces(nGeo)]

                for subFace in subFaces:
                    nFace = (nGeo+1)*(nGeo+2)/2
                    faceNum = faceMap(0) if len(subFace) == nFace else faceMap(1)
                    faceSet = frozenset(subFace)

                    for cnodes, cname in csets_old.items():
                        # Face is not a subset of an existing boundary face
                        if not faceSet.issubset(cnodes):
                            continue

                        # For the first side on the BC, the dict does not exist
                        try:
                            prevSides          = csets_new[cname[0]]
                            prevSides[faceNum] = np.append(prevSides[faceNum], nFaces[faceNum])
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
                elems_new[elemName] = np.append(elems_new[elemName], np.array(subElems).astype(int), axis=0)
            except KeyError:
                elems_new[elemName] = np.array(subElems).astype(int)

    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251

    return mesh


# TODO: FASTER IMPLEMENTATION WOULD ONLY RETURN THE INDICES
def split_hex_to_tets(nodes: list, order: int) -> list:
    """
    Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
    return a list of new tetra element connectivity lists.

    The node numbering convention assumed here:
      (c0, c1, c2, c3, c4, c5, c6, c7)
    is the usual:
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
            return [[nodes[0], nodes[2], nodes[3], nodes[4]],
                    [nodes[0], nodes[1], nodes[2], nodes[4]],
                    [nodes[2], nodes[4], nodes[6], nodes[7]],
                    [nodes[2], nodes[4], nodes[5], nodes[6]],
                    [nodes[1], nodes[2], nodes[4], nodes[5]],
                    [nodes[2], nodes[3], nodes[4], nodes[7]]]
            # ! 2. strategy: 6 tets per box, split hex into two prisms and each prism into 3 tets, periodic but strongly anisotropic
            # c0, c1, c2, c3, c4, c5, c6, c7 = nodes
            # return [[c0, c1, c3, c4],
            #         [c1, c4, c5, c7],
            #         [c1, c3, c4, c7],
            #         [c1, c2, c5, c7],
            #         [c1, c2, c3, c7],
            #         [c2, c5, c6, c7]]
        case 2:
            return [[nodes[0], nodes[2], nodes[3], nodes[4], nodes[24], nodes[10], nodes[11], nodes[16], nodes[26], nodes[20]],
                    [nodes[0], nodes[1], nodes[2], nodes[4], nodes[8] , nodes[9] , nodes[24], nodes[16], nodes[22], nodes[26]],
                    [nodes[2], nodes[4], nodes[6], nodes[7], nodes[26], nodes[25], nodes[18], nodes[23], nodes[15], nodes[14]],
                    [nodes[2], nodes[4], nodes[5], nodes[6], nodes[26], nodes[12], nodes[21], nodes[18], nodes[25], nodes[13]],
                    [nodes[1], nodes[2], nodes[4], nodes[5], nodes[9] , nodes[26], nodes[22], nodes[17], nodes[21], nodes[12]],
                    [nodes[2], nodes[3], nodes[4], nodes[7], nodes[10], nodes[20], nodes[26], nodes[23], nodes[19], nodes[15]]]
        case 4:
            tetra1 = [nodes[0], nodes[2], nodes[3], nodes[4], nodes[80], nodes[88], nodes[82], nodes[14], nodes[15], nodes[16],
                      nodes[19], nodes[18], nodes[17], nodes[32], nodes[33], nodes[34], nodes[100], nodes[124], nodes[102],
                      nodes[47], nodes[52], nodes[45], nodes[98], nodes[122], nodes[114], nodes[108], nodes[101], nodes[118],
                      nodes[44], nodes[51], nodes[48], nodes[84], nodes[85], nodes[81],nodes[109]]
            tetra2 = [nodes[0], nodes[1], nodes[2], nodes[4], nodes[8] , nodes[9] , nodes[10], nodes[11], nodes[12], nodes[13],
                      nodes[82], nodes[88], nodes[80], nodes[32], nodes[33], nodes[34], nodes[63], nodes[70], nodes[65],
                      nodes[100], nodes[124], nodes[102], nodes[62], nodes[66], nodes[69], nodes[99], nodes[107], nodes[120],
                      nodes[98], nodes[122], nodes[114], nodes[87], nodes[83], nodes[86], nodes[106]]
            tetra3 = [nodes[2], nodes[4], nodes[6], nodes[7], nodes[100], nodes[124], nodes[102], nodes[89], nodes[97], nodes[91],
                      nodes[40], nodes[39], nodes[38], nodes[71], nodes[79], nodes[73], nodes[29], nodes[30], nodes[31],
                      nodes[26], nodes[27], nodes[28], nodes[121], nodes[113], nodes[105], nodes[96], nodes[95], nodes[92],
                      nodes[78] , nodes[74] , nodes[77], nodes[116], nodes[123], nodes[104], nodes[112]]
            tetra4 = [nodes[2], nodes[4], nodes[5], nodes[6], nodes[100], nodes[124], nodes[102], nodes[20], nodes[21], nodes[22],
                      nodes[56], nodes[61], nodes[54], nodes[38], nodes[39], nodes[40], nodes[89], nodes[97], nodes[91],
                      nodes[23], nodes[24], nodes[25], nodes[116], nodes[123], nodes[104], nodes[93], nodes[90], nodes[94],
                      nodes[58], nodes[59], nodes[55], nodes[119], nodes[110], nodes[103], nodes[111]]
            tetra5 = [nodes[1], nodes[2], nodes[4], nodes[5], nodes[11], nodes[12], nodes[13], nodes[100], nodes[124], nodes[102],
                      nodes[65], nodes[70], nodes[63], nodes[35], nodes[36], nodes[37], nodes[54], nodes[61], nodes[56],
                      nodes[20], nodes[21], nodes[22], nodes[53], nodes[57], nodes[60], nodes[119], nodes[110], nodes[103],
                      nodes[67], nodes[68], nodes[64], nodes[99], nodes[107], nodes[120], nodes[115]]
            tetra6 = [nodes[2], nodes[3], nodes[4], nodes[7], nodes[14], nodes[15], nodes[16], nodes[47], nodes[52], nodes[45],
                      nodes[102], nodes[124], nodes[100], nodes[71], nodes[79], nodes[73], nodes[41], nodes[42], nodes[43],
                      nodes[29], nodes[30], nodes[31], nodes[75], nodes[72], nodes[76], nodes[50], nodes[49], nodes[46],
                      nodes[121], nodes[113], nodes[105], nodes[108], nodes[101], nodes[118], nodes[117]]

            return [tetra1,tetra2,tetra3,tetra4,tetra5,tetra6]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def tetra_faces(order: int) -> list:
    """
    Given 4 tet corner indices, return the 4 triangular faces as tuples.
    Each face is a triple (n0, n1, n2)
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
            return [np.array([  0,  1,  2,  *range(4,13), *range(31,34)], dtype=int),
                    np.array([  0,  1,  3,  *range(4,7) , *range(16,19),  *reversed(range(13,16)), *range(22,25)], dtype=int),
                    np.array([  0,  2,  3,  *reversed(range(10,13)), *range(19,22),  *reversed(range(13,16)), *range(28,31)], dtype=int),
                    np.array([  1,  2,  3,  *range(7,10), *range(19,22), *reversed(range(16,19)), *range(25,28)], dtype=int)]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


# TODO: FASTER IMPLEMENTATION WOULD ONLY RETURN THE INDICES
def split_hex_to_pyram(nodes: list, center: int, edges: list, order: int) -> list:
    """
    Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
    return a list of new pyramid element connectivity lists.
    """
    match order:
        case 1:
            # Perform the 6-pyramid split of the cube-like cell
            return [tuple((nodes[0], nodes[1], nodes[2], nodes[3], center)),
                    tuple((nodes[0], nodes[4], nodes[5], nodes[1], center)),
                    tuple((nodes[1], nodes[5], nodes[6], nodes[2], center)),
                    tuple((nodes[0], nodes[3], nodes[7], nodes[4], center)),
                    tuple((nodes[4], nodes[7], nodes[6], nodes[5], center)),
                    tuple((nodes[6], nodes[7], nodes[3], nodes[2], center))]
            # 3-pyramid split
            #  return [tuple((nodes[0], nodes[1], nodes[2], nodes[3], nodes[4])),
            #          tuple((nodes[1], nodes[5], nodes[6], nodes[2], nodes[4])),
            #          tuple((nodes[6], nodes[7], nodes[3], nodes[2], nodes[4]))]
        case 2:
            # Perform the 6-pyramid split of the cube-like cell
            return [tuple((nodes[0] , nodes[1] , nodes[2] , nodes[3] , nodes[26], nodes[8] , nodes[9] , nodes[10], nodes[11],
                           edges[0] , edges[1] , edges[3] , edges[2] , nodes[24])),
                    tuple((nodes[0] , nodes[4] , nodes[5] , nodes[1] , nodes[26], nodes[16], nodes[12], nodes[17], nodes[8] ,
                           edges[0] , edges[4] , edges[5] , edges[1] , nodes[22])),
                    tuple((nodes[1] , nodes[5] , nodes[6] , nodes[2] , nodes[26], nodes[17], nodes[13], nodes[18], nodes[9] ,
                           edges[1] , edges[5] , edges[7] , edges[3] , nodes[21])),
                    tuple((nodes[0] , nodes[3] , nodes[7] , nodes[4] , nodes[26], nodes[11], nodes[19], nodes[15], nodes[16],
                           edges[0] , edges[2] , edges[6] , edges[4] , nodes[20])),
                    tuple((nodes[4] , nodes[7] , nodes[6] , nodes[5] , nodes[26], nodes[15], nodes[14], nodes[13], nodes[12],
                           edges[4] , edges[6] , edges[7] , edges[5] , nodes[25])),
                    tuple((nodes[6] , nodes[7] , nodes[3] , nodes[2] , nodes[26], nodes[14], nodes[19], nodes[10], nodes[18],
                           edges[7] , edges[6] , edges[2] , edges[3] , nodes[23]))]
            # 3-pyramid split
            #  return [tuple((nodes[0] , nodes[1] , nodes[2] , nodes[3] , nodes[4], nodes[8] , nodes[9] , nodes[10], nodes[11],
            #                 nodes[16], nodes[22], nodes[26], nodes[20], nodes[24])),
            #          tuple((nodes[1] , nodes[5] , nodes[6] , nodes[2] , nodes[4], nodes[17], nodes[13], nodes[18], nodes[9] ,
            #                 nodes[22], nodes[12], nodes[25], nodes[26], nodes[21])),
            #          tuple((nodes[6] , nodes[7] , nodes[3] , nodes[2] , nodes[4], nodes[14], nodes[19], nodes[10], nodes[18],
            #                 nodes[25], nodes[15], nodes[20], nodes[26], nodes[23]))]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def pyram_faces(order: int) -> list:
    """
    Given the 5 pyramid corner indices, return the 4 triangular faces and 1 quadrilateral face as tuples.
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
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


# TODO: FASTER IMPLEMENTATION WOULD ONLY RETURN THE INDICES
def split_hex_to_prism(nodes: list, order: int) -> list:
    """
    Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
    return a list of new prism element connectivity lists.
    """
    match order:
        case 1:
            return [[nodes[0], nodes[1], nodes[3], nodes[4], nodes[5], nodes[7]],
                    [nodes[1], nodes[2], nodes[3], nodes[5], nodes[6], nodes[7]]]
        case 2:
            #  HEXA: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 24 22 21 23 20 25 26]
            return [[nodes[ 0], nodes[ 1], nodes[ 3], nodes[ 4], nodes[ 5], nodes[ 7], nodes[ 8], nodes[24], nodes[11], nodes[12],
                     nodes[25], nodes[15], nodes[16], nodes[17], nodes[19], nodes[22], nodes[26], nodes[20]],
                    [nodes[ 1], nodes[ 2], nodes[ 3], nodes[ 5], nodes[ 6], nodes[ 7], nodes[ 9], nodes[10], nodes[24],
                     nodes[13], nodes[14], nodes[25], nodes[17], nodes[18], nodes[19], nodes[21], nodes[23], nodes[26]]]
        case 4:
            prism1 =[nodes[ 0], nodes[ 1], nodes[ 3], nodes[ 4], nodes[ 5], nodes[ 7],
                     nodes[ 8], nodes[ 9], nodes[10], nodes[83], nodes[88], nodes[81], nodes[19], nodes[18], nodes[17],#6
                     nodes[20], nodes[21], nodes[22], nodes[90], nodes[97], nodes[92], nodes[31], nodes[30], nodes[29],#15
                     nodes[32], nodes[33], nodes[34], nodes[35], nodes[36], nodes[37], nodes[41], nodes[42], nodes[43],#24
                     nodes[62], nodes[63], nodes[64], nodes[65], nodes[66], nodes[67], nodes[68], nodes[69], nodes[70],#face1:33
                     nodes[99],nodes[101],nodes[105],nodes[103],nodes[122], nodes[117],nodes[123],nodes[115],nodes[124],#face2:42
                     nodes[47], nodes[44], nodes[45], nodes[46], nodes[51], nodes[48], nodes[49], nodes[50], nodes[52],#face3:51
                     nodes[89], nodes[93], nodes[96], #face4 #60
                     nodes[80], nodes[87], nodes[84], #face5 #63
                     nodes[98],nodes[106],nodes[109],nodes[114],nodes[120],nodes[118],nodes[102],nodes[110],nodes[113]] #volume

            prism2 =[nodes[ 1], nodes[ 2], nodes[ 3], nodes[ 5], nodes[ 6], nodes[ 7],
                     nodes[11], nodes[12], nodes[13], nodes[14], nodes[15], nodes[16], nodes[81], nodes[88], nodes[83],#6
                     nodes[23], nodes[24], nodes[25], nodes[26], nodes[27], nodes[28], nodes[92], nodes[97], nodes[90],#15
                     nodes[35], nodes[36], nodes[37], nodes[38], nodes[39], nodes[40], nodes[41], nodes[42], nodes[43],#24
                     nodes[53], nodes[54], nodes[55], nodes[56], nodes[57], nodes[58], nodes[59], nodes[60], nodes[61],#face1
                     nodes[71], nodes[72], nodes[73], nodes[74], nodes[75], nodes[76], nodes[77], nodes[78], nodes[79],#face3
                    nodes[101], nodes[99],nodes[103],nodes[105],nodes[122],nodes[115],nodes[123],nodes[117], nodes[124],#face2
                     nodes[94], nodes[91], nodes[95], #face4
                     nodes[86], nodes[82], nodes[85], #face5
                    nodes[107],nodes[100],nodes[108],nodes[119],nodes[116],nodes[121],nodes[111],nodes[104],nodes[112]] #volume

            return [prism1,prism2]

        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)


@cache
def prism_faces(order: int) -> list:
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
                    np.array([  0, 1, 2, *range(6,15) , *range(63,66)], dtype=int), # z-
                    np.array([  3, 4, 5, *range(15,24), *range(60,63)], dtype=int), # z+
                    # Quadrilateral faces
                    np.array([  0, 1, 4, 3, *range(6,9)  , *range(27,30), 17, 16, 15, 26, 25, 24, *range(33,42)], dtype=int),
                    np.array([  1, 2, 5, 4, *range(9,12) , *range(30,33), 20, 19, 18, 29, 28, 27, *range(42,51)], dtype=int),
                    np.array([  2, 0, 3, 5, *range(12,15), *range(24,27), 23, 22, 21, 32, 31, 30, *range(51,60)], dtype=int)]
        case _:
            print('Order {} not supported for element splitting'.format(order))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
