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
from multiprocessing import Pool, Queue
import plotext as plt
from alive_progress import alive_bar
from typing import Tuple
import threading
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def legendre_gauss_lobatto_nodes(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Return Legendre-Gauss-Lobatto nodes and weights for a given order in 1D
    """
    order -= 1
    # Special cases for small N
    if order == 1:
        return np.array([-1, 1]), np.array([1, 1])

    # Compute the initial guess for the LGL nodes (roots of P'_N)
    nodes = np.cos(np.pi * np.arange(order+1) / order)

    # Initialize the Legendre polynomial and its derivative
    p = np.zeros((order+1, order+1))

    # Iteratively solve for the LGL nodes using Newton's method
    xOld = 2 * np.ones_like(nodes)
    tol = 1e-14
    while np.max(np.abs(nodes - xOld)) > tol:
        xOld = nodes.copy()
        p[:, 0] = 1
        p[:, 1] = nodes
        for k in range(2, order+1):
            p[:, k] = ((2*k-1) * nodes * p[:, k-1] - (k-1) * p[:, k-2]) / k
        nodes -= (nodes * p[:, order] - p[:, order-1]) / (order * (p[:, order]))

    # The LGL nodes
    nodes = np.sort(nodes)

    # Compute the LGL weights
    weights = 2 / (order * (order + 1) * (p[:, order]**2))

    return nodes, weights


def legendre_gauss_nodes(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Return Legendre-Gauss nodes and weights for a given order in 1D
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def barycentric_weights(order: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the barycentric weights for a given node set
        > Algorithm 30, Kopriva
    """
    # Create a difference matrix (x_i - x_j) for all i, j
    diff_matrix = xGP[:, np.newaxis] - xGP[np.newaxis, :]

    # Set the diagonal to 1 to avoid division by zero (diagonal elements will not be used)
    np.fill_diagonal(diff_matrix, 1.0)

    # Compute the product of all differences for each row (excluding the diagonal)
    wBary = np.prod(diff_matrix, axis=1)

    # Take the reciprocal to get the final barycentric weights
    wBary = 1.0 / wBary

    return wBary


def polynomial_derivative_matrix(order: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the polynomial derivative matrix for a given node set
        > Algorithm 37, Kopriva
    """
    wBary = barycentric_weights(order, xGP)
    D     = np.zeros((order, order), dtype=float)

    for iLagrange in range(order):
        for iGP in range(order):
            if iLagrange != iGP:
                D[iGP, iLagrange] = wBary[iLagrange]/(wBary[iGP]*(xGP[iGP]-xGP[iLagrange]))
                D[iGP, iGP      ] = D[iGP, iGP] - D[iGP, iLagrange]

    return D


def lagrange_interpolation_polys(x: float, order: int, xGP: np.ndarray, wBary: np.ndarray) -> np.ndarray:
    """ Computes all Lagrange functions evaluated at position x in [-1;1]
        > Algorithm 34, Kopriva
    """
    # Equal points need special treatment
    L   = np.zeros(order)
    for iGP in range(order):
        if abs(x - xGP[iGP]) < 1.E-14:
            L[iGP] = 1
            return L

    tmp = 0.
    for iGP in range(order):
        L[iGP] = wBary[iGP] / (x-xGP[iGP])
        tmp   += L[iGP]

    # Normalize
    L = L/tmp
    return L


def calc_vandermonde(n_In: int, n_Out: int, wBary_In: np.ndarray, xi_In: np.ndarray, xi_Out: np.ndarray) -> np.ndarray:
    """ Build a 1D Vandermonde matrix using the Lagrange basis functions of degree N_In,
        evaluated at the interpolation points xi_Out
    """
    Vdm = np.zeros((n_Out, n_In))
    for iXI in range(n_Out):
        Vdm[iXI, :] = lagrange_interpolation_polys(xi_Out[iXI], n_In, xi_In, wBary_In)
    return Vdm


# def change_basis_3D(dim1: int, n_In: int, n_Out: int, Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
def change_basis_3D(Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
    """ Interpolate a 3D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
        to another 3D tensor product node positions (number of nodes N_out+1)
        defined by (N_out+1) interpolation point  positions xi_Out(0:N_Out)
        xi is defined in the 1DrefElem xi=[-1,1]
    """
    # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x3D_In)
    x3D_Buf1 = np.tensordot(Vdm, x3D_In, axes=(1, 1))
    x3D_Buf1 = np.moveaxis(x3D_Buf1, 0, 1)  # Correct the shape to (dim1, n_Out, n_In, n_In)

    # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of x3D_Buf1)
    x3D_Buf2 = np.tensordot(Vdm, x3D_Buf1, axes=(1, 2))
    x3D_Buf2 = np.moveaxis(x3D_Buf2, 0, 2)  # Correct the shape to  (dim1, n_Out, n_Out, n_In)

    # Third contraction along the kN_In axis (axis 1 of Vdm, axis 3 of x3D_Buf2)
    x3D_Out  = np.tensordot(Vdm, x3D_Buf2, axes=(1, 3))
    x3D_Out  = np.moveaxis(x3D_Out , 0, 1)  # Correct the shape to (dim1, n_Out, n_Out, n_Out)

    return x3D_Out


def evaluate_jacobian(xGeo_In: np.ndarray, VdmGLtoAP: np.ndarray, D_EqToGL: np.ndarray) -> np.ndarray:
    # Perform tensor contraction for the first derivative (Xi direction)
    dXdXiGL   = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 1))
    dXdXiGL   = np.moveaxis(dXdXiGL  , 0, -3)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)

    # Perform tensor contraction for the second derivative (Eta direction)
    dXdEtaGL  = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 2))
    dXdEtaGL  = np.moveaxis(dXdEtaGL , 0, -3)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)

    # Perform tensor contraction for the third derivative (Zeta direction)
    dXdZetaGL = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 3))
    dXdZetaGL = np.moveaxis(dXdZetaGL, 0, -3)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)

    # Change basis for each direction
    dXdXiAP   = change_basis_3D(VdmGLtoAP, dXdXiGL  )
    dXdEtaAP  = change_basis_3D(VdmGLtoAP, dXdEtaGL )
    dXdZetaAP = change_basis_3D(VdmGLtoAP, dXdZetaGL)

    # Precompute cross products between dXdEtaAP and dXdZetaAP for all points
    cross_eta_zeta = np.cross(dXdEtaAP, dXdZetaAP, axis=0)  # Shape: (3, nGeoRef, nGeoRef, nGeoRef)

    # Fill output Jacobian array
    jacOut = np.einsum('ijkl,ijkl->jkl', dXdXiAP, cross_eta_zeta)

    return jacOut


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


def process_chunk(chunk):
    """Process a chunk of elements by evaluating the Jacobian for each
    """
    chunk_results = []
    for elem in chunk:
        # nodeCoords, nGeoRef, VdmGLtoAP, D_EqToGL = elem
        nodeCoords, _, VdmGLtoAP, D_EqToGL = elem
        jac    = evaluate_jacobian(nodeCoords, VdmGLtoAP, D_EqToGL)
        maxJac = np.max(np.abs(jac))
        minJac = np.min(jac)
        chunk_results.append(minJac / maxJac)
    return chunk_results


def distribute_work(elems, chunk_size):
    """Distribute elements into chunks of a given size
    """
    return [elems[i:i + chunk_size] for i in range(0, len(elems), chunk_size)]


def run_in_parallel(elems, chunk_size=10):
    """Run the element processing in parallel using a specified number of processes
    """
    # Local imports ----------------------------------------
    from pyhope.common.common_vars import np_mtp
    # ------------------------------------------------------

    chunks = distribute_work(elems, chunk_size)
    total_elements = len(elems)
    progress_queue = Queue()

    # Use a separate thread for the progress bar
    progress_thread = threading.Thread(target=update_progress, args=(progress_queue, total_elements))
    progress_thread.start()

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=np_mtp) as pool:
        # Map work across processes in chunks
        results = []
        for chunk_result in pool.imap_unordered(process_chunk, chunks):
            results.extend(chunk_result)
            # Update progress for each processed element in the chunk
            for _ in chunk_result:
                progress_queue.put(1)

    # Wait for the progress bar thread to finish
    progress_thread.join()

    return results


def update_progress(progress_queue, total_elements):
    """ Function to update the progress bar from the queue
    """
    with alive_bar(total_elements, title='│             Processing Elements', length=33) as bar:
        for _ in range(total_elements):
            # Block until we receive a progress update from the queue
            progress_queue.get()
            bar()


def CheckJacobians() -> None:
    # Local imports ----------------------------------------
    from pyhope.common.common_vars import np_mtp
    from pyhope.io.io import LINMAP
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('CHECK JACOBIANS...')
    hopout.sep()

    checkElemJacobians = GetLogical('CheckElemJacobians')
    if not checkElemJacobians:
        return None

    nGeo = mesh_vars.nGeo + 1

    # Compute the equidistant point set used by meshIO
    xEq = np.zeros(nGeo)
    for i in range(nGeo):
        xEq[i] = 2.*float(i)/float(nGeo-1) - 1.
    wBaryEq   = barycentric_weights(nGeo, xEq)

    xGL, _    = legendre_gauss_lobatto_nodes(nGeo)
    DGL       = polynomial_derivative_matrix(nGeo, xGL)
    VdmEqToGL = calc_vandermonde(nGeo, nGeo, wBaryEq, xEq, xGL)
    wbaryGL   = barycentric_weights(nGeo, xGL)

    D_EqToGL  = np.matmul(DGL, VdmEqToGL)

    # Interpolate derivatives on GL (N) to nGeoRef points
    nGeoRef = 3*(nGeo-1)+1
    xAP     = np.zeros(nGeoRef)
    for i in range(nGeoRef):
        xAP[i] = 2. * float(i)/float(nGeoRef-1) - 1.
    VdmGLtoAP = calc_vandermonde(nGeo, nGeoRef, wbaryGL, xGL, xAP)

    # Map all points to tensor product
    elems = mesh_vars.elems
    nodes = mesh_vars.mesh.points

    # Prepare elements for parallel processing
    jacobian_tasks = []

    for iElem, elem in enumerate(elems):
        # Only consider hexahedrons
        if int(elem.type) % 100 != 8:
            continue

        # Get the mapping
        linMap = LINMAP(elem.type, order=mesh_vars.nGeo)
        mapLin = {k: v for v, k in enumerate(linMap)}

        # Fill the NodeCoords
        nodeCoords = np.zeros((nGeo ** 3, 3), dtype=np.float64)
        for iNode, nodeID in enumerate(elem.nodes):
            nodeCoords[mapLin[iNode], :] = nodes[nodeID]

        xGeo = np.zeros((3, nGeo, nGeo, nGeo))
        iNode = 0
        for k in range(nGeo):
            for j in range(nGeo):
                for i in range(nGeo):
                    xGeo[:, i, j, k] = nodeCoords[iNode, :]
                    iNode += 1

        if np_mtp > 0:
            # Add tasks for parallel processing
            jacobian_tasks.append((xGeo, nGeoRef, VdmGLtoAP, D_EqToGL))
        else:
            jac = evaluate_jacobian(xGeo, VdmGLtoAP, D_EqToGL)
            maxJac =  np.max(np.abs(jac))
            minJac =  np.min(       jac)
            jacobian_tasks.append(minJac / maxJac)

    if np_mtp > 0:
        # Run in parallel with a chunk size
        jacs = run_in_parallel(jacobian_tasks, chunk_size=10)
    else:
        jacs = np.array(jacobian_tasks)

    # Plot the histogram of the Jacobians
    plot_histogram(np.array(jacs))
