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
        elemType = GetIntFromStr('ElemType', number=zone)

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

        # We need to define the curves as transfinite curves
        # and set the correct spacing from the parameter file
        for index, line in enumerate(e):
            # We set the number of nodes, so Elems+1
            gmsh.model.geo.mesh.setTransfiniteCurve(line, nElems[edge_to_dir(index, elemType)]+1)

        # Create the curve loop
        el = [None for _ in range(len(faces(elemType)))]
        for index, face in enumerate(faces(elemType)):
            el[index] = gmsh.model.geo.addCurveLoop([math.copysign(e[abs(s)], s) for s in face_to_edge(face, elemType)])

        # Create the surfaces
        s = [None for _ in range(len(faces(elemType)))]
        for index, surface in enumerate(s):
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
    for iVV, vv in enumerate(vvs):
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

    # PyGMSH returns a meshio.mesh datatype
    mesh = pygmsh.geo.Geometry().generate_mesh(order=mesh_vars.nGeo)

    if debugvisu:
        gmsh.fltk.run()

    # Finally done with GMSH, finalize
    gmsh.finalize()

    return mesh
