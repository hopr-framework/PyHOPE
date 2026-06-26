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
from __future__ import annotations
import gc
import re
import sys
from typing import Final, Optional
from collections.abc import Callable
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.common.common_numba import jit, types
from pyhope.mesh.mesh_common import LINMAP
from pyhope.mesh.mesh_common import dir_to_nodes, faces
from pyhope.basis.basis_basis import change_basis_2D
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def init_worker(function : Callable,
                VdmEqToGP: npt.NDArray[np.float64],
                DGP      : npt.NDArray[np.float64],
                weights  : npt.NDArray[np.float64]) -> None:
    """Initializer to set process-local attributes on the worker function
    """
    function.VdmEqToGP = VdmEqToGP
    function.DGP       = DGP
    function.weights   = weights


@jit(types.Tuple((types.float64, types.float64, types.float64))(
    types.float64[:, :, ::1],
    types.float64[:, :, ::1],
    types.float64[:,    ::1]
), nopython=True, cache=True, nogil=True)
def eval_dotprod(dXdetaGP: npt.NDArray[np.float64],
                 dXdxiGP:  npt.NDArray[np.float64],
                 weights:  npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], ...]:
    NSurf0 = -np.sum(weights * (dXdxiGP[1] * dXdetaGP[2] - dXdxiGP[2] * dXdetaGP[1]))
    NSurf1 = -np.sum(weights * (dXdxiGP[2] * dXdetaGP[0] - dXdxiGP[0] * dXdetaGP[2]))
    NSurf2 = -np.sum(weights * (dXdxiGP[0] * dXdetaGP[1] - dXdxiGP[1] * dXdetaGP[0]))
    return NSurf0, NSurf1, NSurf2


def eval_nsurf(XGeo:    npt.NDArray[np.float64],
               Vdm:     npt.NDArray[np.float64],
               DGP:     npt.NDArray[np.float64],
               weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Evaluate the surface integral for normals over a side of an element
    """
    # Change basis to Gauss points
    xGP      = change_basis_2D(Vdm, XGeo)

    # Compute derivatives at all Gauss points
    dXdxiGP  = np.empty_like(xGP)
    dXdetaGP = np.empty_like(xGP)
    DT       = DGP.T
    for k in range(3):
        dXdxiGP[ k] = DGP @ xGP[k]
        dXdetaGP[k] = xGP[k] @ DT

    # Compute the weighted cross product integral
    NSurf0, NSurf1, NSurf2 = eval_dotprod(dXdetaGP, dXdxiGP, weights)

    return np.array((NSurf0, NSurf1, NSurf2), dtype=xGP.dtype)


def check_orientation(ionodes  : npt.NDArray,
                      elemType : int,
                      VdmEqToGP: npt.NDArray[np.float64],
                      DGP      : npt.NDArray[np.float64],
                      weights  : npt.NDArray[np.float64],
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
        indices, doTransp = dir_to_nodes(face, elemType, mesh_vars.nGeo)
        fnodes  = mapnodes[indices] if not doTransp else mapnodes[indices].transpose()
        fpoints = iopoints[fnodes]

        # Calculate high-order surface normal vector via Gauss integration
        nSurf = eval_nsurf(fpoints.transpose(2, 0, 1), VdmEqToGP, DGP, weights)

        # Vector pointing from face center to element center
        fCenter  = fpoints.reshape(-1, 3).mean(axis=0)
        vCenter  = cElem - fCenter

        if np.dot(nSurf, vCenter) > 0.:
            success = False
            sface   = face
            break
    return success, sface


def process_chunk(chunk: tuple) -> list:
    """Process a chunk of elements by checking surface normal orientation
    """
    # Only keep failures to reduce memory and avoid building large arrays of successes
    chunk_results = []
    for elemChunk in chunk:
        iElem, ionodes, elemType = elemChunk
        elem_result = check_orientation(ionodes, elemType,
                                        process_chunk.VdmEqToGP,  # ty: ignore[unresolved-attribute]
                                        process_chunk.DGP,        # ty: ignore[unresolved-attribute]
                                        process_chunk.weights)    # ty: ignore[unresolved-attribute]
        # Append a lightweight sentinel (None) for successes, actual failure list otherwise
        chunk_results.append((elem_result, iElem) if not elem_result[0] else None)
    return chunk_results


def CheckOrient() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.basis.basis_basis import barycentric_weights, legendre_gauss_nodes
    from pyhope.basis.basis_basis import calc_vandermonde, polynomial_derivative_matrix
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
    nGeo: Final[int] = mesh_vars.nGeo

    # Setup mathematical structures identical to CheckWatertight
    xEq:       Final[npt.NDArray[np.float64]] = np.linspace(-1., 1., nGeo+1)
    wBaryEq:   Final[npt.NDArray[np.float64]] = barycentric_weights(nGeo+1, xEq)

    xGP, wGP  = legendre_gauss_nodes(nGeo+1)
    DGP:       Final[npt.NDArray[np.float64]] = polynomial_derivative_matrix(nGeo+1, xGP)
    VdmEqToGP: Final[npt.NDArray[np.float64]] = calc_vandermonde(nGeo+1, nGeo+1, wBaryEq, xEq, xGP)
    weights:   Final[npt.NDArray[np.float64]] = np.outer(wGP, wGP)

    elemNames: Final[dict] = mesh_vars.ELEMTYPE.name
    elemKeys : Final       = mesh_vars.ELEMTYPE.type.keys()
    nElems      = 0
    passedTypes = []

    for elemType in mesh.cells_dict:
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
            res   = run_in_parallel(process_chunk,                                                          # noqa: E251
                                    tasks,                                                                  # noqa: E251
                                    chunk_size = max(1, min(1000, max(10, int(len(tasks)/(40.*np_mtp))))),  # noqa: E251
                                    initializer = init_worker,                                              # noqa: E251
                                    init_args   = (process_chunk, VdmEqToGP, DGP, weights),                 # noqa: E251
                                    ordering   = False,                                                     # noqa: E251
                                   )
        else:
            res   = np.fromiter(((check_orientation(ioelems[iElem - nElems], elemType, VdmEqToGP, DGP, weights), iElem)
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
