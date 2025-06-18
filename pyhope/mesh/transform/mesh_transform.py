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
import importlib.util
import os
import sys
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def CalcStretching(nZones: int, zone: int, nElems: np.ndarray, lEdges: np.ndarray) -> np.ndarray:
    """ Calculate the stretching parameter for meshing the current zone
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import CountOption, GetRealArray
    # ------------------------------------------------------

    # Check if mesh is scaled
    nl0     = CountOption('l0')
    nFactor = CountOption('Factor')

    conditions = {(0     , 0     ): 'constant',     # Non-stretched element arrangement
                  (0     , nZones): 'factor',       # Stretched element arrangement based on factor
                  (nZones, 0     ): 'ratio',        # Stretched element arrangement based on ratio
                  (nZones, nZones): 'combination'   # Stretched element arrangement with a combination of l0 and factor
                 }

    stretchingType = conditions.get((nl0, nFactor), None)

    if stretchingType == 'combination':
        print(hopout.warn('Both l0 and a stretching factor are provided. ' +
                          'The number of elements will be adapted to account for both parameters.'))
    if stretchingType is None:
        hopout.error('Streching parameters not defined properly. Check whether l0 and/or Factor are defined nZone-times.', traceback=True)

    # Calculate the stretching parameter for meshing the current zone
    stretchingHandlers = {
            'constant':    lambda: ([1., 1., 1.], None, None),                         # noqa: E272
            'factor':      lambda: (GetRealArray('Factor', number=zone), None, None),  # noqa: E272
            'ratio':       lambda: (None,                                              # noqa: E272
                                    GetRealArray('l0'    , number=zone),
                                    np.divide(lEdges, np.abs(GetRealArray('l0', number=zone)),
                                              out=np.zeros_like(lEdges), where=GetRealArray('l0', number=zone) != 0)),
            'combination': lambda: (GetRealArray('Factor', number=zone),
                                    GetRealArray('l0'    , number=zone),
                                    np.divide(lEdges, np.abs(GetRealArray('l0', number=zone)),
                                              out=np.zeros_like(lEdges), where=GetRealArray('l0', number=zone) != 0))
    }

    handler = stretchingHandlers.get(stretchingType)
    progFac, l0, dx = handler() if handler else (np.array(()), np.array(()), np.array(()))

    if stretchingType == 'combination':
        for iDim in range(3):
            if np.isclose(progFac[iDim], 0., atol=mesh_vars.tolInternal):
                continue  # Skip if factor is zero, (nElem, l0) given, factor calculated later

            progFac[iDim] = (np.abs(progFac[iDim]))**(np.sign(progFac[iDim]*l0[iDim]))

            if np.isclose(progFac[iDim], 1., atol=mesh_vars.tolInternal):
                continue  # Skip if factor is one, no stretching

            # Calculate the number of elements in the current zone
            nElems[iDim] = max(1, np.rint(np.log(1.-dx[iDim]*(1.-progFac[iDim])) / np.log(progFac[iDim])))
        print(hopout.warn(f'nElems in zone {zone} have been updated to ({nElems[0]}, {nElems[1]},{nElems[2]}).'))

    # Calculate the required factor from ratio or combination input
    if stretchingType in {'ratio', 'combination'}:
        print(hopout.warn(hopout.Colors.WARN + '─'*(46-16) + hopout.Colors.END))
        for iDim in range(3):
            if nElems[iDim] == 1 or dx[iDim] == 0:
                progFac[iDim] = 1.
                continue
            elif nElems[iDim] == 2:
                progFac[iDim] = dx[iDim] - 1.
                continue

            # Start value for Newton iteration
            progFac[iDim] = dx[iDim] / nElems[iDim]

            if np.isclose(progFac[iDim], 1., atol=mesh_vars.tolInternal):
                continue  # Skip iteration if equidistant case

            # Newton iteration
            iter = 0
            while iter < 1000:
                F  = progFac[iDim]**nElems[iDim] + dx[iDim]*(1.-progFac[iDim]) - 1.
                df = nElems[iDim]*progFac[iDim]**(nElems[iDim]-1) - dx[iDim]

                if np.isclose(F, 0., atol=mesh_vars.tolInternal) or np.isclose(df/F, 0., atol=mesh_vars.tolInternal):
                    break

                progFac[iDim] -= F / df
                iter          += 1

            if iter == 1000:
                hopout.error('Newton iteration for computing the stretching function has failed.', traceback=True)

            progFac[iDim] = progFac[iDim]**np.sign(l0[iDim])
            print(hopout.warn(f'New stretching factor [dir {iDim}]: {progFac[iDim]}'))

        print(hopout.warn(hopout.Colors.WARN + '─'*(46-16) + hopout.Colors.END))

    # Return stretching factor
    return progFac


def TransformMesh() -> None:
    # Local imports ----------------------------------------
    from pyhope.config.config import prmfile
    from pyhope.readintools.readintools import CountOption
    from pyhope.readintools.readintools import GetReal, GetRealArray, GetStr
    from pyhope.mesh.mesh_vars import mesh
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    nMeshScale = CountOption('meshScale')
    nMeshTrans = CountOption('meshTrans')
    nMeshRot   = CountOption('meshRot')

    # Read in the mesh post-deformation flag
    meshPostDeform = GetStr('MeshPostDeform') if CountOption('MeshPostDeform') != 0 else 'none'

    # Leave if no transformation is required
    if all(x == 0 for x in [nMeshScale, nMeshTrans, nMeshRot]) and meshPostDeform == 'none':
        return None

    # Start with basic transformations
    hopout.separator()
    hopout.info('TRANSFORM MESH...')
    hopout.sep()

    hopout.routine('Performing basic transformations')
    hopout.sep()

    # Get scaling factor for mesh
    meshScale = GetReal('meshScale')

    # Get translation vector for mesh
    meshTrans = GetRealArray('meshTrans')

    # Get rotation matrix for mesh
    meshRot   = GetRealArray('meshRot')
    meshRot   = np.array(meshRot).reshape(3, 3)
    meshRotC  = GetRealArray('meshRotCenter')

    # Scale mesh
    if meshScale != 1.0:
        mesh.points *= meshScale

    # Translate mesh
    if not np.array_equal(meshTrans, [0.0, 0.0, 0.0]):
        mesh.points += meshTrans

    # Rotate mesh
    if not np.array_equal(meshRot, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        mesh.points = meshRotC + (mesh.points-meshRotC) @ meshRot

    # Exit routine if no further advanced transformation is required
    if meshPostDeform == 'none':
        hopout.sep()
        hopout.info('TRANSFORM MESH DONE!')
        return

    # Continue with advanced transformations
    hopout.sep()
    hopout.routine('Performing advanced transformations')
    hopout.routine('  Template: {}'.format(meshPostDeform))

    # Define locations of the transformation files ( Priority: prmfile folder > CWD > templates )
    DeformLocations = [
        os.path.join(os.path.dirname(prmfile), f'{meshPostDeform}.py'),                # Search folder of parameter file
        os.path.join(os.getcwd(), f'{meshPostDeform}.py'),                             # Search in CWD
        os.path.join(os.path.dirname(__file__), 'templates', f'{meshPostDeform}.py')   # Search in 'templates'
    ]

    # Check if the transformation file exists
    PostDeformMod = None
    for loc in DeformLocations:
        if os.path.exists(loc):
            spec = importlib.util.spec_from_file_location(meshPostDeform, loc)
            # Skip to the next location if spec is None
            if spec is None:
                continue

            PostDeformMod = importlib.util.module_from_spec(spec)
            sys.modules[meshPostDeform] = PostDeformMod
            spec.loader.exec_module(PostDeformMod)

            # Output filename of template
            hopout.routine('     found: {}'.format(loc))

            # Stop once the module is successfully loaded
            break

    # If the transformation file is not found, exit
    if PostDeformMod is None:
        hopout.warning(f'Post Transformation template "{meshPostDeform}" not found!')
        # Print all available default templates for post-deformation
        templist = []
        for file in os.listdir(os.path.join(os.path.dirname(__file__), 'templates')):
            if file.endswith('.py'):
                templist.append(f'  {file[:-3]}')
        hopout.error('Available default transformation templates:' + ','.join(templist))

    # Perform actual post-deformation
    mesh.points = PostDeformMod.PostDeform(mesh.points)

    hopout.sep()
    hopout.info('TRANSFORM MESH DONE!')
