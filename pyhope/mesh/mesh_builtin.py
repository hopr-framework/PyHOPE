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


def MeshCartesian() -> meshio.Mesh:
    # Third-party libraries --------------------------------
    import gmsh
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common import find_index, find_indices, IsDisplay
    from pyhope.io.io_vars import debugvisu
    from pyhope.mesh.mesh_common import edge_to_dir, face_to_corner, face_to_edge, faces
    from pyhope.mesh.mesh_vars import BC
    from pyhope.mesh.transform.mesh_transform import CalcStretching
    from pyhope.meshio.meshio_convert import gmsh_to_meshio
    from pyhope.readintools.readintools import CountOption, GetInt, GetIntArray, GetIntFromStr, GetRealArray, GetStr
    # ------------------------------------------------------

    gmsh.initialize()
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Geometry.Tolerance'         , 1e-12)  # default: 1e-6
        gmsh.option.setNumber('Geometry.MatchMeshTolerance', 1e-09)  # default: 1e-8

    hopout.sep()

    nZones    = GetInt('nZones')
    elemTypes = [int() for _ in range(nZones)]

    offsetp   = 0
    offsets   = 0

    # GMSH only supports mesh elements within a single model
    # > https://gitlab.onelab.info/gmsh/gmsh/-/issues/2836
    gmsh.model.add('Domain')
    gmsh.model.set_current('Domain')
    bcZones = [list() for _ in range(nZones)]

    for zone in range(nZones):
        hopout.routine('Generating zone {}'.format(zone+1))

        # check if corners are given in the input file
        if CountOption('Corner') > 0:
            corners  = GetRealArray( 'Corner'  , number=zone)
        elif CountOption('DX') > 0:
            # get extension of the computational zone
            DX = GetRealArray( 'DX'  , number=zone)

            # read in origin of the zone
            X0 = GetRealArray( 'X0'  , number=zone)

            # reconstruct points from DX and X0 such that all coreners are defined
            corners = np.array((np.array((X0[0],       X0[1],       X0[2]      )),
                                np.array((X0[0]+DX[0], X0[1],       X0[2]      )),
                                np.array((X0[0]+DX[0], X0[1]+DX[1], X0[2]      )),
                                np.array((X0[0],       X0[1]+DX[1], X0[2]      )),
                                np.array((X0[0],       X0[1],       X0[2]+DX[2])),
                                np.array((X0[0]+DX[0], X0[1],       X0[2]+DX[2])),
                                np.array((X0[0]+DX[0], X0[1]+DX[1], X0[2]+DX[2])),
                                np.array((X0[0],       X0[1]+DX[1], X0[2]+DX[2]))))
        else:
            hopout.error('No corners or DX vector given for zone {}'.format(zone+1))

        nElems = GetIntArray(  'nElems'  , number=zone)
        # Store the requested element types
        if CountOption('ElemType') == 1 and zone > 0:
            elemTypes[zone] = elemTypes[0]
        else:
            elemTypes[zone] = GetIntFromStr('ElemType', number=zone)
        # ... but GMSH always builds hexahedral elements
        elemType = 108

        # Create all the corner points
        p = [None for _ in range(len(corners))]
        for index, corner in enumerate(corners):
            p[index] = gmsh.model.geo.addPoint(*corner, tag=offsetp+index+1)

        # Define edge connectivity based on your corner indexing
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        # Connect the corner points
        e = [None for _ in range(12)]
        # First, the plane surface
        for i in range(2):
            for j in range(4):
                e[j + i*4] = gmsh.model.geo.addLine(p[j + i*4], p[(j+1) % 4 + i*4])
        # Then, the connection
        for j in range(4):
            e[j+8] = gmsh.model.geo.addLine(p[j], p[j+4])

        # Extract edge vectors from 'corners' for orientation checking
        edge_vectors = [corners[end] - corners[start] for start, end in edge_pairs]

        # Get dimensions of domain
        gmsh.model.geo.synchronize()
        box    = gmsh.model.get_bounding_box(-1, -1)
        lEdges = np.zeros([3])
        for i in range(3):
            lEdges[i] = np.abs(box[i+3]-box[i])

        # Get streching information of current zone
        stretchType = GetIntArray('StretchType', number=zone)

        # Check which stretching type is used and calculate the required factors
        if np.all(stretchType == 0) and (CountOption('l0') > 0 or CountOption('Factor') > 0):
            # No stretching however check if l0 or factor is gievn in parameter file
            # and assume that the user wants to use the factor stretching by default.
            # Calculate the stretching parameter for meshing the current zone
            stretchType[:] = 1
            # print warning which indicates that default value has been changed
            print(hopout.warn('Default StretchType changed for the current zone since '
                              'Factor or l0 is provided.'))

        # Progression factor stretching or double sided stretching
        stretchFac = np.ndarray([])
        if 1 in stretchType:
            stretchFac = CalcStretching(nZones, zone, nElems, lEdges)

        # Ratio based stretching
        DXmaxToDXmin = np.ndarray([])
        if 2 in stretchType or 3 in stretchType:
            DXmaxToDXmin = GetRealArray('DXmaxToDXmin', number=zone)

        # We need to define the curves as transfinite curves
        # and set the correct spacing from the parameter file
        for index, line in enumerate(e):

            # We set the number of nodes, so Elems+1
            currDir = edge_to_dir(index, elemType)
            stretch_type = stretchType[currDir]

            # Set default values for equidistant elements
            progType = 'Progression'
            progFac = 1.

            # Overwrite default values to consider streching in current zone
            if stretch_type == 3:
                progType = 'Bump'
            if stretch_type == 1:
                progFac = stretchFac[currDir]
            elif stretch_type == 2:
                progFac = -1./(DXmaxToDXmin[currDir] ** (1. / (nElems[currDir] - 1.)))
            elif stretch_type == 3:
                progFac = 1./DXmaxToDXmin[currDir]
            gmsh.model.geo.mesh.setTransfiniteCurve(line, nElems[currDir]+1, progType, (np.sign(edge_vectors[index][currDir]) or 1.) * progFac)

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

        # Assign the volume to a physical zone
        _ = gmsh.model.addPhysicalGroup(3, [zone+1], name='Zone{}'.format(zone+1))

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
        bcs[iBC].name = GetStr(     'BoundaryName', number=iBC).lower()    # noqa: E251
        bcs[iBC].bcid = iBC + 1                                            # noqa: E251
        bcs[iBC].type = GetIntArray('BoundaryType', number=iBC)            # noqa: E251

    nVVs = CountOption('vv')
    if nVVs > 0:
        hopout.sep()
    mesh_vars.vvs = [dict() for _ in range(nVVs)]
    vvs = mesh_vars.vvs
    for iVV, _ in enumerate(vvs):
        vvs[iVV] = dict()
        vvs[iVV]['Dir'] = GetRealArray('vv', number=iVV)

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
                hopout.error('BC "{}" has no periodic vector given, exiting...'.format(iBC + 1), traceback=True)
            else:
                continue

            translation = [1., 0., 0., float(vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'][0]),
                           0., 1., 0., float(vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'][1]),
                           0., 0., 1., float(vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'][2]),
                           0., 0., 0., 1.]

            # Find the opposing side(s)
            # > copy, otherwise we modify bcs
            nbType     = cast(list, copy.copy(bcs[iBC].type))
            nbType[3] *= -1
            nbBCID     = find_index([s.type for s in bcs], nbType)
            # nbSurfID can hold multiple surfaces, depending on the number of zones
            # > find_indices returns all we need!
            nbSurfID   = [s+1 for s in find_indices(bcIndex, nbBCID+1)]

            # Connect positive to negative side
            try:
                gmsh.model.mesh.setPeriodic(2, nbSurfID, surfID, translation)
                hopout.routine('Generated periodicity constraint with vector {}'.format(
                    vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir']))

            # If the number of sides do not match, we cannot impose periodicity
            # > Leave it out here and assume we can sort it out in ConnectMesh
            except Exception as e:
                print(hopout.warn(' No GMSH periodicity with vector {}'.format(
                    vvs[int(cast(np.ndarray, bcs[iBC].type)[3])-1]['Dir'])))
                continue

    if len(vvs) > 0:
        hopout.sep()

    # To generate connect the generated cells, we can simply set
    gmsh.option.setNumber('Mesh.RecombineAll'  , 1)
    gmsh.option.setNumber('Mesh.Recombine3DAll', 1)
    gmsh.option.setNumber('Geometry.AutoCoherence', 2)
    gmsh.model.mesh.recombine()
    # Force Gmsh to output all mesh elements
    gmsh.option.setNumber('Mesh.SaveAll', 1)

    gmsh.model.mesh.generate(3)

    # Set the element order
    # > This needs to be executed after generate_mesh, see
    # > https://github.com/nschloe/pygmsh/issues/515#issuecomment-1020106499
    gmsh.model.mesh.setOrder(mesh_vars.nGeo)
    gmsh.model.geo.synchronize()

    if debugvisu and IsDisplay():
        gmsh.fltk.run()

    # Convert Gmsh object to meshio object
    mesh = gmsh_to_meshio(gmsh)

    # Finally done with GMSH, finalize
    gmsh.finalize()

    # Run garbage collector to release memory
    gc.collect()

    # Store the element types
    mesh_vars.nZones    = nZones
    mesh_vars.elemTypes = elemTypes

    return mesh
