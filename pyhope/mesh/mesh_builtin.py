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
import copy
import gc
import math
import sys
import traceback
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import gmsh
import meshio
import numpy as np
import pygmsh
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def MeshCartesian() -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common import find_index, find_indices
    from pyhope.io.io_vars import debugvisu
    from pyhope.mesh.mesh_common import edge_to_dir, face_to_corner, face_to_edge, faces
    from pyhope.mesh.mesh_vars import BC
    from pyhope.mesh.mesh_transform import CalcStretching
    from pyhope.readintools.readintools import CountOption, GetInt, GetIntFromStr, GetIntArray, GetRealArray, GetStr
    # ------------------------------------------------------

    gmsh.initialize()
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Geometry.Tolerance'         , 1e-12)  # default: 1e-6
        gmsh.option.setNumber('Geometry.MatchMeshTolerance', 1e-09)  # default: 1e-8

    hopout.sep()

    nZones = GetInt('nZones')

    offsetp  = 0
    offsets  = 0

    # GMSH only supports mesh elements within a single model
    # > https://gitlab.onelab.info/gmsh/gmsh/-/issues/2836
    gmsh.model.add('Domain')
    gmsh.model.set_current('Domain')
    bcZones = [list() for _ in range(nZones)]

    for zone in range(nZones):
        hopout.routine('Generating zone {}'.format(zone+1))

        corners  = GetRealArray( 'Corner'  , number=zone)
        nElems   = GetIntArray(  'nElems'  , number=zone)
        elemType = 108  # GMSH always builds hexahedral elements

        # Create all the corner points
        p = [None for _ in range(len(corners))]
        for index, corner in enumerate(corners):
            p[index] = gmsh.model.geo.addPoint(*corner, tag=offsetp+index+1)

        # Connect the corner points
        e = [None for _ in range(12)]
        # First, the plane surface
        for i in range(2):
            for j in range(4):
                e[j + i*4] = gmsh.model.geo.addLine(p[j + i*4], p[(j+1) % 4 + i*4])
        # Then, the connection
        for j in range(4):
            e[j+8] = gmsh.model.geo.addLine(p[j], p[j+4])

        # Get dimensions of domain
        gmsh.model.geo.synchronize()
        box    = gmsh.model.get_bounding_box(-1, -1)
        lEdges = np.zeros([3])
        for i in range(3):
            lEdges[i] = np.abs(box[i+3]-box[i])

        # Calculate the stretching parameter for meshing the current zone
        progFac = CalcStretching(nZones, zone, nElems, lEdges)

        # We need to define the curves as transfinite curves
        # and set the correct spacing from the parameter file
        for index, line in enumerate(e):
            # We set the number of nodes, so Elems+1
            currDir = edge_to_dir(index, elemType)
            gmsh.model.geo.mesh.setTransfiniteCurve(line, nElems[currDir[0]]+1, 'Progression', currDir[1]*progFac[currDir[0]])

        # Create the curve loop
        el = [None for _ in range(len(faces(elemType)))]
        for index, face in enumerate(faces(elemType)):
            el[index] = gmsh.model.geo.addCurveLoop([math.copysign(e[abs(s)], s) for s in face_to_edge(face, elemType)])

        # Create the surfaces
        s = [None for _ in range(len(faces(elemType)))]
        for index, _ in enumerate(s):
            s[index] = gmsh.model.geo.addPlaneSurface([el[index]], tag=offsets+index+1)

        # We need to define the surfaces as transfinite surface
        for index, face in enumerate(faces(elemType)):
            gmsh.model.geo.mesh.setTransfiniteSurface(offsets+index+1, face, [p[s] for s in face_to_corner(face, elemType)])
            gmsh.model.geo.mesh.setRecombine(2, 1)

        # Create the surface loop
        gmsh.model.geo.addSurfaceLoop([s for s in s], zone+1)

        gmsh.model.geo.synchronize()

        # Create the volume
        gmsh.model.geo.addVolume([zone+1], zone+1)

        # We need to define the volume as transfinite volume
        gmsh.model.geo.mesh.setTransfiniteVolume(zone+1)
        gmsh.model.geo.mesh.setRecombine(3, 1)

        # Calculate all offsets
        offsetp += len(corners)
        offsets += len(faces(elemType))

        # Read the BCs for the zone
        # > Need to wait with defining physical boundaries until all zones are created
        bcZones[zone] = [int(s) for s in GetIntArray('BCIndex')]

    # At this point, we can create a "Physical Group" corresponding
    # to the boundaries. This requires a synchronize call!
    gmsh.model.geo.synchronize()

    hopout.sep()
    hopout.routine('Setting boundary conditions')
    hopout.sep()
    nBCs = CountOption('BoundaryName')
    mesh_vars.bcs = [BC() for _ in range(nBCs)]
    bcs = mesh_vars.bcs

    for iBC, bc in enumerate(bcs):
        # bcs[iBC].update(name = GetStr(     'BoundaryName', number=iBC),  # noqa: E251
        #                 bcid = iBC + 1,                                  # noqa: E251
        #                 type = GetIntArray('BoundaryType', number=iBC))  # noqa: E251
        bcs[iBC].name = GetStr(     'BoundaryName', number=iBC)  # noqa: E251
        bcs[iBC].bcid = iBC + 1                                  # noqa: E251
        bcs[iBC].type = GetIntArray('BoundaryType', number=iBC)  # noqa: E251

    nVVs = CountOption('vv')
    mesh_vars.vvs = [dict() for _ in range(nVVs)]
    vvs = mesh_vars.vvs
    for iVV, _ in enumerate(vvs):
        vvs[iVV] = dict()
        vvs[iVV]['Dir'] = GetRealArray('vv', number=iVV)
    if len(vvs) > 0:
        hopout.sep()

    # Flatten the BC array, the surface numbering follows from the 2-D ordering
    bcIndex = [item for row in bcZones for item in row]

    bc = [None for _ in range(max(bcIndex))]
    for iBC in range(max(bcIndex)):
        # if mesh_vars.bcs[iBC-1] is None:
        # if 'Name' not in bcs[iBC]:
        if bcs[iBC] is None:
            continue

        # Format [dim of group, list, name)
        # > Here, we return ALL surfaces on the BC, irrespective of the zone
        surfID  = [s+1 for s in find_indices(bcIndex, iBC+1)]
        bc[iBC] = gmsh.model.addPhysicalGroup(2, surfID, name=cast(str, bcs[iBC].name))

        # For periodic sides, we need to impose the periodicity constraint
        if cast(np.ndarray, bcs[iBC].type)[0] == 1:
            # > Periodicity transform is provided as a 4x4 affine transformation matrix, given by row
            # > Rotation matrix [columns 0-2], translation vector [column 3], bottom row [0, 0, 0, 1]

            # Only define the positive translation
            if cast(np.ndarray, bcs[iBC].type)[3] > 0:
                pass
            elif cast(np.ndarray, bcs[iBC].type)[3] == 0:
                hopout.warning('BC "{}" has no periodic vector given, exiting...'.format(iBC + 1))
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            else:
                continue

            hopout.routine('Generated periodicity constraint with vector {}'.format(
                vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir']))

            translation = [1., 0., 0., float(vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'][0]),
                           0., 1., 0., float(vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'][1]),
                           0., 0., 1., float(vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'][2]),
                           0., 0., 0., 1.]

            # Find the opposing side(s)
            # > copy, otherwise we modify bcs
            nbType     = copy.copy(bcs[iBC].type)
            nbType[3] *= -1
            nbBCID     = find_index([s.type for s in bcs], nbType)
            # nbSurfID can hold multiple surfaces, depending on the number of zones
            # > find_indices returns all we need!
            nbSurfID   = [s+1 for s in find_indices(bcIndex, nbBCID+1)]

            # Connect positive to negative side
            gmsh.model.mesh.setPeriodic(2, nbSurfID, surfID, translation)

    # To generate connect the generated cells, we can simply set
    gmsh.option.setNumber('Mesh.RecombineAll'  , 1)
    gmsh.option.setNumber('Mesh.Recombine3DAll', 1)
    gmsh.option.setNumber('Geometry.AutoCoherence', 2)
    gmsh.model.mesh.recombine()
    # Force Gmsh to output all mesh elements
    gmsh.option.setNumber('Mesh.SaveAll', 1)

    # Set the element order
    # > Technically, this is only required in generate_mesh but let's be precise here
    gmsh.model.mesh.setOrder(mesh_vars.nGeo)
    gmsh.model.geo.synchronize()

    if debugvisu:
        gmsh.fltk.run()

    # PyGMSH returns a meshio.mesh datatype
    mesh = pygmsh.geo.Geometry().generate_mesh(order=mesh_vars.nGeo)

    # # Calculate the offset for the quad cells
    # offset    = 0
    # for elems in mesh.cells:
    #     if any(sub in elems.type for sub in {'vertex', 'line'}):
    #         offset += len(elems.data)
    #
    # # Remove 0D/1D entities
    # elems  = []
    # csets  = {}
    # for key, val in enumerate(mesh.cells):
    #     if any(sub in val.type for sub in {'vertex', 'line'}):
    #         for cset in mesh.cell_sets.items():
    #             cset_new = [(cast(np.ndarray, s) - offset) if s is not None else None for s in cset[1][key+1:]]
    #             csets.update({cset[0]: cset_new})
    #     else:
    #         elems.append(val)
    #
    # mesh = meshio.Mesh(points    = mesh.points,  # noqa: E251
    #                    cells     = elems,        # noqa: E251
    #                    cell_sets = csets)        # noqa: E251
    # del elems, csets

    # Split elements if simplex elements are requested
    elemType = GetIntFromStr('ElemType')
    if elemType % 100 != 8:
        # FIXME: Currently not supported
        if mesh_vars.nGeo > 1:
            hopout.warning('Non-hexahedral elements are not supported for nGeo > 1, exiting...')
            sys.exit(1)

        mesh = MeshChangeElemType(mesh, elemType)

    # Finally done with GMSH, finalize
    gmsh.finalize()

    # Run garbage collector to release memory
    gc.collect()

    return mesh


def MeshChangeElemType(mesh: meshio.Mesh, elemType: int) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------
    # Copy original points
    points    = mesh.points.copy()
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    # Prepare new cell blocks and new cell_sets
    elems_new = {}
    csets_new = {}
    elemName  = ELEMTYPE.inam[elemType][0]

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
            if elems_old[blockID].type != 'quad':
                continue

            # Sort them as a set for membership checks
            for face in block:
                nodes = mesh.cells_dict[elems_old[blockID].type][face - offset]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    # Set up the element splitting function
    elemSplitter = { 104: (split_hex_to_tets , tetra_faces),
                     105: (split_hex_to_pyram, pyram_faces),
                     106: (split_hex_to_prism, prism_faces)
                   }
    split, faces = elemSplitter.get(elemType, (None, None))

    if split is None or faces is None:
        hopout.warning('Element type {} not supported for splitting'.format(elemType))
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    nPoints  = len(points)
    nFaces   = np.zeros(2)
    faceType = ['triangle', 'quad']

    faceMaper = { 104: lambda x: 0,
                  105: lambda x: 0 if x == 0 else 1,
                  106: lambda x: 0 if x == 0 else 1}
    faceMap   = faceMaper.get(elemType, None)

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        if ctype != 'hexahedron':
            continue

        # Hex block: Split each element
        for elem in cdata:
            # Pyramids need a center node
            if elemType == 105:
                center   = np.mean(points[elem], axis=0)
                center   = np.expand_dims(center, axis=0)
                points   = np.append(points, center, axis=0)
                subElems = split(elem, nPoints)
                nPoints += 1
            else:
                subElems = split(elem)

            for subElem in subElems:
                for subFace in faces(subElem):
                    faceNum = faceMap(0) if len(subFace) == 3 else faceMap(1)
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
                            prevSides          = [np.empty((0,), dtype=int) for _ in range(2)]
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


def split_hex_to_tets(nodes: list) -> list:
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
    c0, c1, c2, c3, c4, c5, c6, c7 = nodes

    # Perform the 6-tet split of the cube-like cell
    # 1. strategy: 6 tets per box, all tets have same volume and angle, not periodic but isotropic
    return [[c0, c2, c3, c4],
            [c0, c1, c2, c4],
            [c2, c4, c6, c7],
            [c2, c4, c5, c6],
            [c1, c2, c4, c5],
            [c2, c3, c4, c7]]
    # ! 2. strategy: 6 tets per box, split hex into two prisms and each prism into 3 tets, periodic but strongly anisotropic
    # return [[c0, c1, c3, c4],
    #         [c1, c4, c5, c7],
    #         [c1, c3, c4, c7],
    #         [c1, c2, c5, c7],
    #         [c1, c2, c3, c7],
    #         [c2, c5, c6, c7]]


def tetra_faces(nodes: list) -> list:
    """
    Given 4 tet corner indices, return the 4 triangular faces as tuples.
    Each face is a triple (n0, n1, n2)
    """
    t0, t1, t2, t3 = nodes
    return [tuple((t0, t1, t2)),
            tuple((t0, t1, t3)),
            tuple((t0, t2, t3)),
            tuple((t1, t2, t3))]


def split_hex_to_pyram(nodes: list, center: int) -> list:
    """
    Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
    return a list of new pyramid element connectivity lists.
    """
    c0, c1, c2, c3, c4, c5, c6, c7 = nodes

    # Perform the 6-pyramid split of the cube-like cell
    return [tuple((c0, c1, c2, c3, center)),
            tuple((c0, c4, c5, c1, center)),
            tuple((c1, c5, c6, c2, center)),
            tuple((c0, c3, c7, c4, center)),
            tuple((c4, c7, c6, c5, center)),
            tuple((c6, c7, c3, c2, center))]


def pyram_faces(nodes: list) -> list:
    """
    Given the 5 pyramid corner indices, return the 4 triangular faces and 1 quadrilateral face as tuples.
    """
    p0, p1, p2, p3, p4 = nodes
    return [# Triangular faces  # noqa: E261
            tuple((p0, p1, p4)),
            tuple((p1, p2, p4)),
            tuple((p2, p3, p4)),
            tuple((p3, p0, p4)),
            # Quadrilateral face
            tuple((p0, p1, p2, p3))]


def split_hex_to_prism(nodes: list) -> list:
    """
    Given the 8 corner node indices of a single hexahedral element (indexed 0..7),
    return a list of new prism element connectivity lists.
    """
    c0, c1, c2, c3, c4, c5, c6, c7 = nodes

    # Perform the 2-prism split of the cube-like cell
    return [[c0, c3, c4, c1, c2, c5],
            [c3, c7, c4, c2, c6, c5]]


def prism_faces(nodes: list) -> list:
    """
    Given the 6 prism corner indices, return the 2 triangular and 3 quadrilateral faces as tuples.
    """
    t0, t1, t2, t3, t4, t5 = nodes
    return [# Triangular faces  # noqa: E261
            tuple(sorted((t0, t1, t2))),
            tuple(sorted((t3, t4, t5))),
            # Quadrilateral faces
            tuple(sorted((t0, t1, t4, t3))),
            tuple(sorted((t1, t2, t5, t4))),
            tuple(sorted((t2, t0, t3, t5)))]
