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

    def CalcStretching() -> np.ndarray:
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------

        # Initialize arrays
        progFac = np.zeros(3)
        l0      = np.zeros(3)
        dx      = np.zeros(3)

        # Calculate the stretching parameter for meshing the current zone
        if stretchingType == 'constant':
            progFac = [1., 1., 1.]
        elif stretchingType == 'factor':
            progFac = GetRealArray('Factor', number = zone)
        elif stretchingType == 'ratio':
            l0 = GetRealArray('l0', number = zone)
            with np.errstate(divide='ignore', invalid='ignore'):
                dx = lEdges/np.abs(l0) # l/l0
            # dx = np.where(l0 > 1.E-12, lEdges / np.abs(l0), 0)
            if np.any(dx < 1.-1.E-12):
                hopout.warning('stretching error, length l0 longer than grid region, in direction.')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
        elif stretchingType == 'combination':
            progFac = GetRealArray('Factor', number = zone)
            l0      = GetRealArray('l0', number = zone)
            with np.errstate(divide='ignore', invalid='ignore'):
                dx = lEdges/np.abs(l0) # l/l0
            for iDim in range(3):
                if not np.abs(progFac[iDim]) < 1.E-12: # fac=0 , (nElem,l0) given, fac calculated
                    progFac[iDim]=(np.abs(progFac[iDim]))**(np.sign(progFac[iDim]*l0[iDim])) # sign for direction
                    if progFac[iDim] != 1.:
                        nElems[iDim]=np.rint(np.log(1.-dx[iDim]*(1.-progFac[iDim])) / np.log(progFac[iDim])) #nearest Integer
                    if nElems[iDim] < 1:
                        nElems[iDim] = 1
            print(hopout.warn("nElems in zone {} have been updated to (/{}, {}, {}/).".format(zone,nElems[0],nElems[1],nElems[2])))

        # Calculate the required factor from ratio or combination input
        if stretchingType == 'ratio' or stretchingType == 'combination':
            for iDim in range(3):
                if nElems[iDim] == 1:
                    progFac[iDim] = 1.0
                elif nElems[iDim] == 2:
                    progFac[iDim] = dx[iDim]-1.
                else: #nElems > 2
                    if np.isinf(dx[iDim]):
                        progFac[iDim] = 1.
                    else:
                        progFac[iDim] = dx[iDim]/nElems[iDim]  #start value for Newton iteration
                        if np.abs(progFac[iDim]-1.) > 1.E-12:  # NEWTON iteration, only if not equidistant case
                            F    = 1.
                            dF   = 1.
                            iter = 0
                            while (np.abs(F) > 1.E-12) and (np.abs(F/dF) > 1.E-12) and (iter < 1000):
                                F  = progFac[iDim]**nElems[iDim] + dx[iDim]*(1.-progFac[iDim]) -1. # non-linear function
                                dF = nElems[iDim]*progFac[iDim]**(nElems[iDim]-1) -dx[iDim]  #dF/dfac
                                progFac[iDim] = progFac[iDim] - F/dF
                                iter=iter+1
                            if iter > 1000:
                                break # 'Newton iteration for computing the stretching function has failed.'
                            progFac[iDim] = progFac[iDim]**np.sign(l0[iDim]) # sign for direction
                    print(f'   -stretching factor in dir {iDim} is now {progFac[iDim]}')

        # Return stretching factor
        return progFac

    gmsh.initialize()
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Geometry.Tolerance'         , 1e-12)  # default: 1e-6
        gmsh.option.setNumber('Geometry.MatchMeshTolerance', 1e-09)  # default: 1e-8

    hopout.sep()

    nZones = GetInt('nZones')

    # Check if mesh is scaled
    nl0      = CountOption('l0')
    nFactor = CountOption('Factor')

    if nl0 == 0 and nFactor == 0:
        # Non-stretched element arrangement
        stretchingType = 'constant'
    elif nl0 == 0 and nFactor == nZones:
        # Stretched element arrangement based on factor
        stretchingType = 'factor'
    elif nl0 == nZones and nFactor == 0:
        # Stretched element arrangement based on ratio
        stretchingType = 'ratio'
    elif nl0 == nZones and nFactor == nZones:
        # Stretched element arrangement with a combination of l0 and factor
        stretchingType = 'combination'
        print(hopout.warn("Both l0 and a stretching factor are provided. The number of elements will be adapted to account for both parameters."))
    else:
        hopout.warning('Streching parameters not defined properly. Check whether l0 and/or Factor are defined nZone-times.')
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

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

        # Get dimensions of domain
        gmsh.model.geo.synchronize()
        box    = gmsh.model.get_bounding_box(-1,-1)
        lEdges = np.zeros([3])
        for i in range(3):
            lEdges[i] = np.abs(box[i+3]-box[i])

        # Calculate the stretching parameter for meshing the current zone
        progFac = CalcStretching()

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
