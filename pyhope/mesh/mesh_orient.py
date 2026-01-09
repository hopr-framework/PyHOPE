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
import gc
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
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.common.common_numba import jit, types
from pyhope.mesh.mesh_common import LINMAP
from pyhope.mesh.mesh_common import dir_to_nodes, faces
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


@jit((types.float64)(types.float64[:, :, ::1], types.float64[::1]), nopython=True, cache=True, nogil=True)
def eval_dotprod(fpoints, nVecFace) -> np.float64:
    # nVecFace = nVecFace / np.linalg.norm(nVecFace)
    # > Manually compute norm
    nVecFace = nVecFace / np.sqrt(nVecFace[0]*nVecFace[0] + nVecFace[1]*nVecFace[1] + nVecFace[2]*nVecFace[2])

    vec1 = fpoints[-1, 0, :] - fpoints[0, 0, :]
    vec2 = fpoints[0, -1, :] - fpoints[0, 0, :]

    # normal = np.cross(vec1, vec2)
    # > Manually compute cross product
    normal = np.empty_like(vec1)
    normal[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    normal[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    normal[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    # Dot product and check if normal points outwards
    # > Manually compute dot product
    dotprod = nVecFace[0]*normal[0] + nVecFace[1]*normal[1] + nVecFace[2]*normal[2]
    return dotprod


def check_orientation(ionodes : np.ndarray,
                      elemType: int,
                     ) -> tuple[bool, Optional[str]]:
    """ Check the orientation of the surface normals
    """
    mapLin   = LINMAP(elemType, order=mesh_vars.nGeo)
    iopoints = mesh_vars.mesh.points
    mapnodes = ionodes[mapLin]
    points   = iopoints[mapnodes]

    # Center of element
    cElem    = points.reshape(-1, 3).mean(axis=0)

    success  = True
    sface    = None

    for face in faces(elemType):
        # Center of face
        indices, doTransp = dir_to_nodes(face, elemType, mesh_vars.nGeo)
        fnodes  = mapnodes[indices] if not doTransp else mapnodes[indices].transpose()
        fpoints = iopoints[fnodes]

        # Tangent and normal vectors
        nVecFace = cElem - fpoints.reshape(-1, 3).mean(axis=0)
        # nVecFace = nVecFace / np.linalg.norm(nVecFace)
        # nVecFace = nVecFace / np.sqrt(np.dot(nVecFace, nVecFace))
        # > Manually compute dot product
        dotprod = eval_dotprod(fpoints, nVecFace)

        if dotprod < 0.:
            success = False
            sface   = face
            break
    return success, sface


def process_chunk(chunk) -> list:
    """Process a chunk of elements by checking surface normal orientation
    """
    # Only keep failures to reduce memory and avoid building large arrays of successes
    chunk_results = []
    for elemChunk in chunk:
        iElem, ionodes, elemType = elemChunk
        elem_result = check_orientation(ionodes, elemType)
        # Append a lightweight sentinel (None) for successes, actual failure list otherwise
        chunk_results.append((elem_result, iElem) if not elem_result[0] else None)
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
            res   = run_in_parallel(process_chunk,
                                    tasks,
                                    chunk_size=max(1, min(1000, max(10, int(len(tasks)/(40.*np_mtp))))),
                                   )
        else:
            res   = np.fromiter(((check_orientation(ioelems[iElem - nElems], elemType), iElem)
                                  for iElem in range(nElems, nElems + nIOElems)), dtype=object)

        if len(res) > 0 and not np.all([success for (success, _), _ in res]):
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

    # Run garbage collector to release memory
    gc.collect()
