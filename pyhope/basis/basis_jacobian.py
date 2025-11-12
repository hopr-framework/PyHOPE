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
import plotext as plt
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
from pyhope.basis.basis_basis import evaluate_jacobian, evaluate_jacobian_simplex
from pyhope.mesh.mesh_common import NDOFS_ELEM
# ==================================================================================================================================

# Define element types for simplex and hexahedron separately
SIMPLEX_TYPES: Final = {4, 5, 6}
HEX_TYPE:      Final = 8


def evaluate_jacobian_dispatch(nodeCoords, VdmGLtoAP, D_EqToGL, elem_type):
    if elem_type in SIMPLEX_TYPES:
        return evaluate_jacobian_simplex(nodeCoords, VdmGLtoAP, D_EqToGL)
    elif elem_type == HEX_TYPE:
        return evaluate_jacobian(        nodeCoords, VdmGLtoAP, D_EqToGL)
    else:
        raise ValueError(f"Unsupported element type {elem_type}")


def plot_histogram(data: np.ndarray) -> None:
    """ Plot a histogram of all Jacobians
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.output.output import STD_LENGTH
    # ------------------------------------------------------

    ticks = ['│<0.0      │',
             '│ 0.0-0.1  │',
             '│ 0.1-0.2  │',
             '│ 0.2-0.3  │',
             '│ 0.3-0.4  │',
             '│ 0.4-0.5  │',
             '│ 0.5-0.6  │',
             '│ 0.6-0.7  │',
             '│ 0.7-0.8  │',
             '│ 0.8-0.9  │',
             # '│ 0.9-0.99 │',
             '│>0.9-1.0  │']

    # Define the bins for categorizing jacobians
    # bins     = [ -np.inf, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, np.inf]
    bins     = [ -np.inf, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,       np.inf]

    # Use np.histogram to count jacobians in the defined bins
    count, _ = np.histogram(data, bins=bins)

    # Setup plot
    hopout.sep()
    hopout.info('Scaled Jacobians')
    hopout.separator(18)
    plt.simple_bar(ticks, count, width=STD_LENGTH)
    plt.show()
    hopout.separator(18)


# Use Pool initializer to attach process-local data to the worker function
def init_worker(function, VdmGLtoAP, D_EqToGL) -> None:
    """Initializer to set process-local attributes on the worker function
    """
    function.VdmGLtoAP = VdmGLtoAP
    function.D_EqToGL  = D_EqToGL


def process_chunk(chunk) -> list[np.ndarray]:
    """Process a chunk of elements by evaluating the Jacobian for each
    """
    chunk_results = []
    for elem in chunk:
        nodeCoords, elem_type = elem
        jac = evaluate_jacobian_dispatch(nodeCoords,
                                         process_chunk.VdmGLtoAP[elem_type],  # pyright: ignore[reportFunctionMemberAccess] # ty: ignore[unresolved-attribute]
                                         process_chunk.D_EqToGL[ elem_type],  # pyright: ignore[reportFunctionMemberAccess] # ty: ignore[unresolved-attribute]
                                         elem_type)
        # INFO: ALTERNATIVE VERSION, CACHING VDM, D
        # nodeCoords, evaluate_jacobian = elem
        # jac    = evaluate_jacobian(nodeCoords)
        # maxJac = np.max(np.abs(jac))
        # minJac = np.min(jac)
        # chunk_results.append(minJac / maxJac)
        chunk_results.append(jac.min() / np.abs(jac).max())
    return chunk_results


def CheckJacobians() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.basis.basis_basis import equi_nodes_tetra, polynomial_derivative_matrix_tetra
    from pyhope.basis.basis_basis import equi_nodes_prism, polynomial_derivative_matrix_prism
    from pyhope.basis.basis_basis import equi_nodes_pyram, polynomial_derivative_matrix_pyram
    from pyhope.basis.basis_basis import barycentric_weights
    from pyhope.basis.basis_basis import polynomial_derivative_matrix
    from pyhope.basis.basis_basis import calc_vandermonde
    from pyhope.basis.basis_basis import legendre_gauss_lobatto_nodes
    # INFO: ALTERNATIVE VERSION, CACHING VDM, D
    # from pyhope.basis.basis_basis import JacobianEvaluator
    from pyhope.common.common_parallel import run_in_parallel
    from pyhope.common.common_vars import np_mtp
    from pyhope.readintools.readintools import GetLogical
    from pyhope.mesh.mesh_common import LINTEN
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('CHECK JACOBIANS...')
    hopout.sep()

    checkElemJacobians = GetLogical('CheckElemJacobians')
    if not checkElemJacobians:
        return None

    # Map all points to tensor product
    nGeo:      Final[int]        = mesh_vars.nGeo + 1
    elems:     Final[list]       = mesh_vars.elems
    nodes:     Final[np.ndarray] = mesh_vars.mesh.points
    elemBases: Final[set]        = set([e.type % 100 for e in elems])

    # Compute the equidistant point set used by meshIO
    xEq_fn      = {4: lambda: equi_nodes_tetra(nGeo),                                          # Tetrahedron
                   5: lambda: equi_nodes_pyram(nGeo),                                          # Pyramid
                   6: lambda: equi_nodes_prism(nGeo),                                          # Wedge / Prism
                   8: lambda: np.linspace(-1, 1, num=nGeo, dtype=np.float64)}                  # Hexahedron
    xEq         = {b: xEq_fn[b]()        for b in elemBases}  # noqa: E272
    wBaryEq     = barycentric_weights(nGeo, np.linspace(-1, 1, num=nGeo, dtype=np.float64))

    # For hexahedron only
    xGL = DGL = VdmEqToGL = wbaryGL = np.array([])
    if 8 in elemBases:
        xGL, _    = legendre_gauss_lobatto_nodes(nGeo)
        DGL       = polynomial_derivative_matrix(nGeo, xGL)
        VdmEqToGL = calc_vandermonde(nGeo, nGeo, wBaryEq, xEq[8], xGL)
        wbaryGL   = barycentric_weights(nGeo, xGL)

    # Interpolate derivatives on GL (N) to nGeoRef points
    D_EqToGL_fn  = {4: lambda: polynomial_derivative_matrix_tetra(nGeo, xEq[4]),              # Tetrahedron
                    5: lambda: polynomial_derivative_matrix_pyram(nGeo, xEq[5]),              # Pyramid
                    6: lambda: polynomial_derivative_matrix_prism(nGeo, xEq[6]),              # Wedge / Prism
                    8: lambda: np.matmul(DGL, VdmEqToGL)}                                     # Hexahedron
    D_EqToGL     = {b: D_EqToGL_fn[b]()  for b in elemBases}  # noqa: E272

    # Interpolate derivatives on GL (N) to nGeoRef points
    nGeoRef      = 3*(nGeo-1)+1
    xAP_fn       = {4: lambda: equi_nodes_tetra(nGeoRef),                                     # Tetrahedron
                    5: lambda: equi_nodes_pyram(nGeoRef),                                     # Pyramid
                    6: lambda: equi_nodes_prism(nGeoRef),                                     # Wedge / Prism
                    8: lambda: np.linspace(-1, 1, num=nGeoRef, dtype=np.float64)}             # Hexahedron
    VdmGLtoAP_fn = {4: lambda: 0,  # calc_vandermonde_tetra(nGeo, nGeoRef, xEq[4], xAP[4]),   # Tetrahedron
                    5: lambda: 0,  # calc_vandermonde_tetra(nGeo, nGeoRef, xEq[5], xAP[5]),   # Pyramid
                    6: lambda: 0,  # calc_vandermonde_prism(nGeo, nGeoRef, xEq[6], xAP[6]),   # Wedge / Prism
                    8: lambda: calc_vandermonde      (nGeo, nGeoRef, wbaryGL, xGL, xAP[8])}   # Hexahedron    # noqa: E211
    xAP          = {b: xAP_fn[b]()       for b in elemBases}  # noqa: E272
    VdmGLtoAP    = {b: VdmGLtoAP_fn[b]() for b in elemBases}
    # INFO: ALTERNATIVE VERSION, CACHING VDM, D
    # evaluate_jacobian = JacobianEvaluator(VdmGLtoAP, D_EqToGL).evaluate_jacobian

    # Prepare elements for parallel processing
    tasks = []

    # Pre-compute LINTEN mappings for all element types
    linCache  = {}
    elemOrder = 100 if mesh_vars.nGeo == 1 else 200
    elemTypes = tuple([s + elemOrder for s in elemBases])
    for elemType in elemTypes:
        try:
            _, mapLin = LINTEN(elemType, order=mesh_vars.nGeo)
            mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))
            linCache[elemType] = mapLin
        # Only hexahedrons supported for specific nGeo
        except ValueError:
            pass

    for elem in elems:
        elemType = elem.type
        elemBase = int(elemType) % 100

        # Get the mapping
        mapLin = linCache[elemType]

        # Fill the NodeCoords
        nodeCoords         = np.empty((nGeo ** 3, 3), dtype=np.float64)
        nodeCoords[mapLin] = nodes[elem.nodes]

        # Hexahedron
        if elemBase == 8:
            xGeo = nodeCoords[:nGeo**3].reshape((nGeo, nGeo, nGeo, 3), order='F').transpose(3, 0, 1, 2)
        # All other elem types
        else:
            xGeo = nodeCoords[:NDOFS_ELEM(elemType, nGeo - 1)].transpose(1, 0)

        if np_mtp > 0:
            # Add tasks for parallel processing
            tasks.append((xGeo, elemBase))
            # INFO: ALTERNATIVE VERSION, CACHING VDM, D
            # tasks.append((xGeo, evaluate_jacobian))
        else:
            jac = evaluate_jacobian_dispatch(xGeo,
                                             VdmGLtoAP[elemBase],
                                             D_EqToGL[ elemBase],
                                             elemBase)
            # INFO: ALTERNATIVE VERSION, CACHING VDM, D
            # jac = evaluate_jacobian(xGeo)
            # maxJac =  np.max(np.abs(jac))
            # minJac =  np.min(       jac)
            # tasks.append(minJac / maxJac)
            tasks.append(jac.min() / np.abs(jac).max())

    if np_mtp > 0:
        # Run in parallel with a chunk size
        # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
        jacs = run_in_parallel(process_chunk,
                               tasks,
                               chunk_size  = max(1, min(1000, max(10, int(len(tasks)/(40.*np_mtp))))),  # noqa: E251
                               initializer = init_worker,                                               # noqa: E251
                               init_args   = (process_chunk, VdmGLtoAP, D_EqToGL))                      # noqa: E251
    else:
        jacs = np.array(tasks)

    # Plot the histogram of the Jacobians
    if len(jacs) > 0:
        plot_histogram(np.array(jacs))

    # Append the Jacobians to the elements
    jacIter = iter(jacs)
    for elem in elems:
        elemType = elem.type
        elem.jacobian = next(jacIter)
