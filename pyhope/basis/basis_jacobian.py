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
import threading
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


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
    # Local imports ----------------------------------------
    from pyhope.basis.basis_basis import evaluate_jacobian
    # ------------------------------------------------------
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
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.basis.basis_basis import barycentric_weights, polynomial_derivative_matrix, calc_vandermonde
    from pyhope.basis.basis_basis import legendre_gauss_lobatto_nodes, evaluate_jacobian
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

    for _, elem in enumerate(elems):
        # Only consider hexahedrons
        if int(elem.type) % 100 != 8:
            continue

        # Get the mapping
        linMap = LINTEN(elem.type, order=mesh_vars.nGeo)
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
