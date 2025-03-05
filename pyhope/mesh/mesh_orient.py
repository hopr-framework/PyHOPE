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
import re
import sys
from typing import Final, Optional
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
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.mesh.mesh_common import LINMAP
from pyhope.mesh.mesh_common import dir_to_nodes, faces
# ==================================================================================================================================


def check_orientation(ionodes : np.ndarray,
                      elemType: int) -> tuple[bool, Optional[str]]:
    """ Check the orientation of the surface normals
    """
    mapLin   = LINMAP(elemType, order=mesh_vars.nGeo)
    nodes    = ionodes[mapLin]
    iopoints = mesh_vars.mesh.points
    points   = iopoints[ionodes[mapLin]]

    # Center of element
    cElem = np.mean(points, axis=(0, 1, 2))

    success = True
    sface   = None
    for face in faces(elemType):
        # Center of face
        indices, doTransp = dir_to_nodes(face, elemType, mesh_vars.nGeo)
        fnodes  = nodes[indices]
        if doTransp:
            fnodes = fnodes.transpose()
        fpoints = iopoints[fnodes]
        # cFace  = fpoints.mean(axis=tuple(range(fpoints.ndim - 1)))
        cFace  = np.mean(fpoints, axis=(0, 1))

        # Tangent and normal vectors
        nVecFace = cElem - cFace
        # nVecFace = nVecFace / np.linalg.norm(nVecFace)
        nVecFace = nVecFace / np.sqrt(np.dot(nVecFace, nVecFace))
        vec1 = fpoints[-1, 0, :] - fpoints[0, 0, :]
        vec2 = fpoints[0, -1, :] - fpoints[0, 0, :]
        normal = np.cross(vec1, vec2)
        # normal /= np.linalg.norm(normal)
        normal /= np.sqrt(np.dot(normal, normal))

        # Dot product and check if normal points outwards
        dotprod = np.dot(nVecFace, normal)
        if dotprod < 0:
            success = False
            sface   = face
            break
    return success, sface


def process_chunk(chunk) -> np.ndarray:
    """Process a chunk of elements by checking surface normal orientation
    """
    chunk_results = np.fromiter(((check_orientation(ionodes, elemType), iElem)
                                  for iElem, ionodes, elemType in chunk), dtype=object)
    return chunk_results


def OrientMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_parallel import run_in_parallel
    from pyhope.common.common_vars import np_mtp
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.routine('Ensuring normals point outward')
    hopout.sep()

    checkSurfaceNormals = GetLogical('CheckSurfaceNormals')
    if not checkSurfaceNormals:
        return None

    mesh = mesh_vars.mesh

    elemNames: Final[dict] = mesh_vars.ELEMTYPE.name
    elemKeys : Final       = mesh_vars.ELEMTYPE.type.keys()
    nElems      = 0
    passedTypes = []

    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in elemKeys):
            continue

        # Only consider hexahedrons
        if 'hexahedron' not in elemType:
            passedTypes.append(elemType)
            continue

        # Get the elements
        ioelems  = mesh.get_cells_type(elemType)
        nIOElems = ioelems.shape[0]

        if isinstance(elemType, str):
            elemType = elemNames[elemType]

        # Prepare elements for parallel processing
        if np_mtp > 0:
            tasks = tuple((iElem, ioelems[iElem - nElems], elemType)
                           for iElem in range(nElems, nElems + nIOElems))
            # Run in parallel with a chunk size
            # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
            res   = run_in_parallel(process_chunk, tasks, chunk_size=max(1, min(1000, max(10, int(len(tasks)/(40.*np_mtp))))))
        else:
            res   = np.fromiter(((check_orientation(ioelems[iElem - nElems], elemType), iElem)
                                  for iElem in range(nElems, nElems + nIOElems)), dtype=object)

        if not np.all([success for (success, _), _ in res]):
            failed_elems = [(iElem + 1, face) for (success, face), iElem in res if not success]
            for iElem, face in failed_elems:
                print(hopout.warn(f'> Element {iElem}, Side {face}'))
            sys.exit(1)

        # Add to nElems
        nElems += nIOElems

    # Warn if we passed any element types
    if len(passedTypes) > 0:
        print(hopout.warn('Ignored element type{}: {}'.format('s' if len(passedTypes) > 1 else '',
                                                              [re.sub(r"\d+$", "", s) for s in passedTypes])))
