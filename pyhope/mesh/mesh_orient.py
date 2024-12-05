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
from typing import Union
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def check_orientation(ionodes : np.ndarray,
                      elemType: Union[str, int],
                      mapLin  : np.ndarray,
                      iopoints: np.ndarray) -> tuple[bool, Union[None, str]]:
    # Local imports ----------------------------------------
    from pyhope.mesh.mesh_common import dir_to_nodes, faces
    # ------------------------------------------------------
    nodes  = ionodes[mapLin]
    points = iopoints[ionodes[mapLin]]

    # Center of element
    cElem = points.reshape(-1, points.shape[-1]).sum(axis=0) / np.prod(ionodes.shape)

    success = True
    sface   = None
    for face in faces(elemType):
        # Center of face
        fnodes = dir_to_nodes(face, elemType, nodes)
        fpoints = iopoints[fnodes]
        match face:
            case 'y-' | 'x+' | 'z+':
                fpoints = fpoints.transpose(1, 0, 2)
        cFace = fpoints.reshape(-1, fpoints.shape[-1]).sum(axis=0) / np.prod(fnodes.shape)

        # Tangent and normal vectors
        nVecFace = cElem - cFace
        nVecFace = nVecFace / np.linalg.norm(nVecFace)
        vec1 = fpoints[-1, 0, :] - fpoints[0, 0, :]
        vec2 = fpoints[0, -1, :] - fpoints[0, 0, :]
        normal = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))

        # Dot product and check if normal points outwards
        dotprod = np.dot(nVecFace, normal)
        if dotprod < 0:
            success = False
            sface   = face
            break
    return success, sface


def process_chunk(chunk) -> list:
    """Process a chunk of elements by checking surface normal orientation."""

    chunk_results = []
    for elem_data in chunk:
        iElem, ionodes, elemType, mapLin, iopoints = elem_data
        success, sface = check_orientation(ionodes, elemType, mapLin, iopoints)
        chunk_results.append((success, iElem, sface))
    return chunk_results


def OrientMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_common import LINMAP
    from pyhope.common.common_parallel import run_in_parallel
    from pyhope.common.common_vars import np_mtp
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.sep()
    hopout.routine('Eliminating duplicate points')

    # Eliminate duplicate points
    mesh_vars.mesh.points, inverseIndices = np.unique(mesh_vars.mesh.points, axis=0, return_inverse=True)

    # Update the mesh
    for cell in mesh_vars.mesh.cells:
        # Map the old indices to the new ones
        # cell.data = np.vectorize(lambda idx: inverseIndices[idx])(cell.data)
        # Efficiently map all indices in one operation
        cell.data = inverseIndices[cell.data]

    hopout.sep()
    hopout.routine('Checking if surface normal vectors point outwards')
    hopout.sep()

    checkSurfaceNormals = GetLogical('CheckSurfaceNormals')
    if not checkSurfaceNormals:
        return None

    mesh   = mesh_vars.mesh
    nElems = 0

    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        # Get the elements
        ioelems  = mesh.get_cells_type(elemType)
        nIOElems = ioelems.shape[0]

        if isinstance(elemType, str):
            elemType = mesh_vars.ELEMTYPE.name[elemType]
        mapLin = LINMAP(elemType, order=mesh_vars.nGeo)

        # Prepare elements for parallel processing
        tasks = []

        # Check the element orientation
        if np_mtp > 0:
            for iElem in range(nElems, nElems+nIOElems):
                tasks.append((iElem, ioelems[iElem], elemType, mapLin, mesh_vars.mesh.points))
        else:
            for iElem in range(nElems, nElems+nIOElems):
                success, sface = check_orientation(ioelems[iElem], elemType, mapLin, mesh_vars.mesh.points)
                tasks.append((success, iElem, sface))

        if np_mtp > 0:
            # Run in parallel with a chunk size
            res = run_in_parallel(process_chunk, tasks, chunk_size=10)
        else:
            res = np.array(tasks)

        for success, iElem, face in res:
            if not success:
                hopout.warning('Surface normals are not pointing outwards, exiting...')
                print(hopout.warn(f'> Element {iElem+1}, Side {face}'))  # noqa: E501
                sys.exit(1)

        # Add to nElems
        nElems += nIOElems
