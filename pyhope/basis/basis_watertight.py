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
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.basis.basis_basis import change_basis_2D
from pyhope.mesh.mesh_common import face_to_nodes
# ==================================================================================================================================


def eval_nsurf(XGeo: np.ndarray, Vdm: np.ndarray, DGP: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ Evaluate the surface integral for normals over a side of an element
    """
    # Change basis to Gauss points
    xGP      = change_basis_2D(Vdm, XGeo)

    # Compute derivatives at all Gauss points
    dXdxiGP  = np.tensordot(DGP, xGP, axes=(1, 1)).transpose(1, 0, 2)  # Shape: (3, N_GP+1, N_GP+1)
    # dXdxiGP  = np.moveaxis(dXdxiGP , 0, 1).reshape(3, -1)              # Flatten for cross computation (slower)
    dXdxiGP  = dXdxiGP .reshape(3, -1)                                 # Flatten for cross computation

    dXdetaGP = np.tensordot(DGP, xGP, axes=(1, 2)).transpose(1, 0, 2)  # Shape: (3, N_GP+1, N_GP+1)
    # dXdetaGP = np.moveaxis(dXdetaGP, 0, 1).reshape(3, -1)              # Flatten for cross computation (slower)
    dXdetaGP = dXdetaGP.reshape(3, -1)                                 # Flatten for cross computation

    # Compute the cross product at each Gauss point
    VDMSize  = Vdm.shape[-1]
    nVec     = np.cross(dXdxiGP, dXdetaGP, axis=0)  # Shape: (3, N_GP*N_GP)
    nVec     = nVec.reshape(3, VDMSize, VDMSize)    # Reshape to (3, N_GP+1, N_GP+1)

    # Compute the weighted normals
    nVecW    = nVec * weights                       # Broadcast weights to shape (3, N_GP+1, N_GP+1)

    # Integrate over the Gauss points
    NSurf    = -np.sum(nVecW, axis=(1, 2))          # Sum over the last two axes
    return NSurf


def check_sides(elem,
                # points   : np.ndarray,
                VdmEqToGP: np.ndarray,
                DGP      : np.ndarray,
                weights  : np.ndarray,
                # sides    : list
                ) -> list[bool | int | np.ndarray]:
    results = []
    points  = mesh_vars.mesh.points
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides
    nGeo    = mesh_vars.nGeo

    # Calculate the cube root of the element volume
    elemTol  = np.cbrt(np.prod(np.ptp(points[elem.nodes], axis=0)))
    elemType = elem.type

    for SideID in elem.sides:
        # TODO: THIS IS CURRENTLY IGNORED, MEANING WE CHECK EVERY CONNECTION DOUBLE
        # if checked[SideID]:
        #     continue

        side   = sides[SideID]

        # Only connected sides and not small mortar sides
        # > Small mortar sides connect to big mortar side, so we will never match
        if side.connection is None or side.sideType < 0:
            continue

        # Big mortar side
        elif side.connection < 0:
            mortarType = abs(side.connection)
            # INFO: This should be faster but I could not confirm the speedup in practice
            # nSurf   = eval_nsurf(np.moveaxis( points[  nodes], 2, 0), VdmEqToGP, DGP, weights)
            # nSurf   = eval_nsurf(np.transpose(np.take(points,   nodes, axis=0), axes=(2, 0, 1)), VdmEqToGP, DGP, weights)
            idx     = elem.nodes[face_to_nodes(side.face, elemType, nGeo)]
            nSurf   = eval_nsurf(np.transpose(points[idx], axes=(2, 0, 1)), VdmEqToGP, DGP, weights)

            # Calculate the L2 norm of the side and take the maximum
            sideTol = np.linalg.norm(nSurf, ord=2)
            tol     = np.max((elemTol, sideTol)) * mesh_vars.tolInternal
            # checked[SideID] = True

            # Mortar sides are the following virtual sides
            nMortar = 4 if mortarType == 1 else 2
            nnbSurf = np.zeros((3,), dtype=float)
            for mortarSide in range(nMortar):
                # Get the matching side
                # INFO: This should be faster but I could not confirm the speedup in practice
                # nnbSurf += eval_nsurf(np.moveaxis(points[nbnodes], 2, 0), VdmEqToGP, DGP, weights)
                nbside   = sides[sides[SideID + mortarSide + 1].connection]
                nbelem   = elems[nbside.elemID]
                idx      = nbelem.nodes[face_to_nodes(nbside.face, nbelem.type, nGeo)]
                nnbSurf += eval_nsurf(np.transpose(points[idx], axes=(2, 0, 1)), VdmEqToGP, DGP, weights)
                # checked[nbside] = True

            # Check if side normals are within tolerance
            nSurfErr = np.sum(np.abs(nnbSurf + nSurf))
            success  = nSurfErr < tol

        # Internal side
        elif side.connection >= 0:
            # Ignore the virtual mortar sides
            if side.locMortar is not None:
                continue

            # INFO: This should be faster but I could not confirm the speedup in practice
            # nSurf   = eval_nsurf(np.moveaxis( points[  nodes], 2, 0), VdmEqToGP, DGP, weights)
            idx     = elem.nodes[face_to_nodes(side.face, elemType, nGeo)]
            nSurf   = eval_nsurf(np.transpose(points[idx]), VdmEqToGP, DGP, weights)

            # Calculate the L2 norm of the side and take the maximum
            sideTol = np.linalg.norm(nSurf, ord=2)
            tol     = np.max((elemTol, sideTol)) * mesh_vars.tolInternal
            # checked[SideID] = True

            # Connected side
            nbside  = sides[side.connection]
            nbelem  = elems[nbside.elemID]
            # INFO: This should be faster but I could not confirm the speedup in practice
            # nnbSurf = eval_nsurf(np.moveaxis(points[nbnodes], 2, 0), VdmEqToGP, DGP, weights)
            idx     = nbelem.nodes[face_to_nodes(nbside.face, nbelem.type, nGeo)]
            nnbSurf = eval_nsurf(np.transpose(points[idx]), VdmEqToGP, DGP, weights)
            # checked[nbside] = True

            # Check if side normals are within tolerance
            nSurfErr = np.sum(np.abs(nnbSurf + nSurf))
            success  = nSurfErr < tol

        else:
            continue

        results.append((success, SideID, nSurf, nnbSurf, nSurfErr, tol))
    return results


def process_chunk(chunk) -> np.ndarray:
    """Process a chunk of elements by checking surface normal orientation
    """
    chunk_results    = np.empty(len(chunk), dtype=object)
    # elem, VdmEqToGP, DGP, weights = elem_data
    chunk_results[:] = [check_sides(*elem_data) for elem_data in chunk]
    return chunk_results


def CheckWatertight() -> None:
    """ Check if the mesh is watertight
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.basis.basis_basis import barycentric_weights, legendre_gauss_nodes
    from pyhope.basis.basis_basis import calc_vandermonde, polynomial_derivative_matrix
    from pyhope.common.common_parallel import run_in_parallel
    from pyhope.common.common_vars import np_mtp
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('CHECK WATERTIGHTNESS...')
    hopout.sep()

    checkWatertightness = GetLogical('CheckWatertightness')
    if not checkWatertightness:
        return None

    nGeo:      Final[int] = mesh_vars.nGeo

    # Compute the equidistant point set used by meshIO
    xEq:       Final[np.ndarray] = np.linspace(-1., 1., nGeo+1)
    wBaryEq:   Final[np.ndarray] = barycentric_weights(nGeo+1, xEq)

    xGP, wGP  = legendre_gauss_nodes(nGeo+1)
    DGP:       Final[np.ndarray] = polynomial_derivative_matrix(nGeo+1, xGP)
    VdmEqToGP: Final[np.ndarray] = calc_vandermonde(nGeo+1, nGeo+1, wBaryEq, xEq, xGP)

    # Compute the weights
    weights:   Final[np.ndarray] = np.outer(wGP, wGP)                   # Shape: (N_GP+1, N_GP+1)

    # Check all sides
    elems:     Final[list] = mesh_vars.elems
    sides:     Final[list] = mesh_vars.sides
    mesh:      Final             = mesh_vars.mesh
    points:    Final[np.ndarray] = mesh.points
    # points    = mesh_vars.mesh.points
    # checked   = np.zeros((len(sides)), dtype=bool)

    # Only consider hexahedrons
    if any(e.type % 100 != 8 for e in elems):
        elemTypes = list(set([e.type for e in elems if e.type % 100 != 8]))
        print(hopout.warn('Ignored element type: {}'.format(
            [re.sub(r"\d+$", "", mesh_vars.ELEMTYPE.inam[e][0]) for e in elemTypes]
        )))
        return

    # Prepare elements for parallel processing
    if np_mtp > 0:
        tasks  = tuple((elem, VdmEqToGP, DGP, weights)
                        for elem in elems)
        # Run in parallel with a chunk size
        # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
        res    = run_in_parallel(process_chunk, tasks, chunk_size=max(1, min(1000, max(10, int(len(tasks)/(40.*np_mtp))))))
    else:
        res    = np.empty(len(elems), dtype=object)
        res[:] = [check_sides(elem, VdmEqToGP, DGP, weights) for elem in elems]

    results = tuple(tuple(result for r in res for result in r if not bool(result[0]))
)
    if len(results) > 0:
        nconn = len(tuple(tuple(result for r in res for result in r)))

        for result in results:
            # Unpack the results
            side    = sides[result[1]]
            elem    = elems[side.elemID]
            nbside  = sides[side.connection]
            nbelem  = elems[nbside.elemID]

            nSurf, nbnSurf, nSurfErr, tol = result[2], result[3], result[4], result[5]

            nodes   =   elem.nodes[face_to_nodes(  side.face,   elem.type, nGeo)]
            nbnodes = nbelem.nodes[face_to_nodes(nbside.face, nbelem.type, nGeo)]

            print()
            # Check if side is oriented inwards
            errStr  =      'Side is oriented inwards!' if nSurfErr < 0 \
                  else 'Surface normals are not within tolerance {:9.6e} > {:9.6e}'.format(nSurfErr, tol)
            print(hopout.warn(errStr, length=len(errStr)+16))

            # Print the information
            strLen  = max(len(str(side.sideID+1)), len(str(nbside.sideID+1)))
            print(hopout.warn(f'> Element {  elem.elemID+1:>{strLen}}, Side {  side.face}, Side {  side.sideID+1:>{strLen}}'))  # noqa: E501
            print(hopout.warn('> Normal vector: [' + ' '.join('{:12.3f}'.format(s) for s in   nSurf) + ']'))                    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[ 0,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[ 0, -1]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[-1,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[-1, -1]]) + ']'))    # noqa: E271
            # print()
            print(hopout.warn(f'> Element {nbelem.elemID+1:>{strLen}}, Side {nbside.face}, Side {nbside.sideID+1:>{strLen}}'))  # noqa: E501
            print(hopout.warn('> Normal vector: [' + ' '.join('{:12.3f}'.format(s) for s in nbnSurf) + ']'))                    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[ 0,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[ 0, -1]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[-1,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[-1, -1]]) + ']'))    # noqa: E271

        hopout.error(f'Watertightness check failed for {len(results)} / {nconn} connections!')
