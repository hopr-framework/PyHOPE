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
from typing import Final, Optional, cast
from collections.abc import Iterable
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
from pyhope.common.common_numba import NUMBA_AVAILABLE
if typing.TYPE_CHECKING or NUMBA_AVAILABLE:
    import numpy.typing as npt
    from pyhope.mesh.mesh_vars import ELEM
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.basis.basis_watertight import eval_nsurf
from pyhope.mesh.mesh_common import face_to_nodes, faces
# ==================================================================================================================================


def check_orientation(elem       : ELEM,
                      VdmEqToGP  : npt.NDArray[np.float64],
                      DGP        : npt.NDArray[np.float64],
                      weights    : npt.NDArray[np.float64],
                      failed_only: bool = False,
                     ) -> Optional[list[tuple]]:
    """ Check the orientation of the surface normals
    """
    points   = mesh_vars.mesh.points
    nGeo     = mesh_vars.nGeo
    elemType = elem.type

    if elemType % 10 != 8:
        return None

    # Center of element
    cElem   = points[elem.nodes].reshape(-1, 3).mean(axis=0)
    results = None

    # Fall back to geometry topology faces if sides array is not initialized
    sides = getattr(mesh_vars, 'sides', None)
    sideIter = ((sides[s].face, s) for s in elem.sides) if (elem.sides is not None and sides is not None) else \
               ((f, None) for f in faces(elemType))

    for face, SideID in sideIter:
        idx     = cast(np.ndarray, elem.nodes)[face_to_nodes(face, elemType, nGeo)]
        fpoints = points[idx]

        # Calculate high-order surface normal vector via Gauss integration
        nSurf   = eval_nsurf(fpoints.transpose(2, 0, 1), VdmEqToGP, DGP, weights)

        # Vector pointing from face center to element center
        fCenter = fpoints.reshape(-1, 3).mean(axis=0)
        vCenter = cElem - fCenter

        success = np.dot(nSurf, vCenter) <= 0.

        # If requested, only return errors
        if failed_only and success:
            continue

        # Lazily initialize results on first failure
        if results is None:
            results = []
        results.append((success, elem.elemID, face, SideID))

    # Avoid creating empty lists on elem_results
    if results is None:
        return None if failed_only else []

    return results


def process_chunk(chunk: tuple) -> list:
    """Process a chunk of elements by checking surface normal orientation
    """
    # Only keep failures to reduce memory and avoid building large arrays of successes
    chunk_results = []
    for elem in chunk:
        elem_result = check_orientation(elem,
                                        process_chunk.VdmEqToGP,  # ty: ignore[unresolved-attribute]
                                        process_chunk.DGP,        # ty: ignore[unresolved-attribute]
                                        process_chunk.weights,    # ty: ignore[unresolved-attribute]
                                        failed_only=True)
        chunk_results.append(elem_result)
    return chunk_results


def CheckOrient() -> None:
    """ Check if element surface normals point outwards
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.basis.basis_basis import barycentric_weights, legendre_gauss_nodes
    from pyhope.basis.basis_basis import calc_vandermonde, polynomial_derivative_matrix
    from pyhope.basis.basis_watertight import init_worker
    from pyhope.common.common_parallel import run_in_parallel
    from pyhope.common.common_vars import np_mtp
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('CHECK NORMALS POINTING OUTWARDS...')
    hopout.sep()

    checkSurfaceNormals = GetLogical('CheckSurfaceNormals')
    if not checkSurfaceNormals:
        return None

    nGeo:  Final[int]  = mesh_vars.nGeo
    elems: Final[list] = mesh_vars.elems

    # Setup mathematical structures identical to CheckWatertight
    xEq:       Final[npt.NDArray[np.float64]] = np.linspace(-1., 1., nGeo+1)
    wBaryEq:   Final[npt.NDArray[np.float64]] = barycentric_weights(nGeo+1, xEq)

    xGP, wGP  = legendre_gauss_nodes(nGeo+1)
    DGP:       Final[npt.NDArray[np.float64]] = polynomial_derivative_matrix(nGeo+1, xGP)
    VdmEqToGP: Final[npt.NDArray[np.float64]] = calc_vandermonde(nGeo+1, nGeo+1, wBaryEq, xEq, xGP)
    weights:   Final[npt.NDArray[np.float64]] = np.outer(wGP, wGP)

    # Prepare elements for parallel processing
    if np_mtp > 0:
        # Run in parallel with a chunk size
        # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
        res = run_in_parallel(process_chunk,
                              elems,
                              chunk_size  = max(1, min(1000, max(10, int(len(elems)/(40.*np_mtp))))),  # noqa: E251
                              initializer = init_worker,                                               # noqa: E251
                              init_args   = (process_chunk, VdmEqToGP, DGP, weights),                  # noqa: E251
                              ordering    = False,                                                     # noqa: E251
                             )
    else:
        res = [check_orientation(elem, VdmEqToGP, DGP, weights, failed_only=True) for elem in elems]

    # Flatten results while filtering empty items
    results = tuple(result for elem_results in res if isinstance(elem_results, Iterable) and elem_results is not None
                           for result       in elem_results)  # noqa: E272

    if len(results) > 0:
        for result in cast(tuple[tuple], results):
            _, elemID, face, sideID = result
            hopout.info('')
            print(hopout.warn(f'Side is oriented inwards! Element {elemID + 1}, Face {face}, Side {sideID + 1}'))

        hopout.error(f'Surface normals check failed for {len(results)} / {len(elems)} elements!')

    # Warn if we passed any element types
    if any(e.type % 10 != 8 for e in elems):
        elemTypes = list({e.type for e in elems if e.type % 10 != 8})
        print(hopout.warn('Ignored element type: {}'.format([re.sub(r"\d+$", "", mesh_vars.ELEMTYPE.inam[e][0]) for e in elemTypes])))  # ruff: ignore[line-too-long]

    # Run garbage collector to release memory
    gc.collect()
