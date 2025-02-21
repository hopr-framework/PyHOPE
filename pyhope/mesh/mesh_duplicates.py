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
import copy
import gc
import sys
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def EliminateDuplicates() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.connect.connect import find_bc_index
    # ------------------------------------------------------
    hopout.routine('Removing duplicate points')

    bcs    = mesh_vars.bcs
    vvs    = mesh_vars.vvs
    mesh   = mesh_vars.mesh

    # Find the mapping to the (N-1)-dim elements
    csetMap = {key: [s for s in range(len(cset)) if cset[s] is not None and np.size(cset[s]) > 0]
                       for key, cset in mesh.cell_sets.items()}

    # Create new periodic nodes per (original node, boundary) pair
    # > Use a dictionary mapping (node, bc_key) --> new node index
    node_bc_translation = {}

    for bc_key, cset in mesh.cell_sets.items():
        # Find the matching boundary condition
        bcID = find_bc_index(bcs, bc_key)
        if bcID is None:
            hopout.warning(f'Could not find BC {bc_key} in list, exiting...')
            sys.exit(1)

        # Only process periodic boundaries in the positive direction
        if bcs[bcID].type[0] != 1 or bcs[bcID].type[3] < 0:
            continue

        iVV = bcs[bcID].type[3]
        VV  = vvs[np.abs(iVV)-1]['Dir'] * np.sign(iVV)

        for iMap in csetMap[bc_key]:
            # Only process 2D faces (quad or triangle)
            if not any(s in list(mesh.cells_dict)[iMap] for s in ['quad', 'triangle']):
                continue

            iBCsides = np.array(cset[iMap]).astype(int)
            mapFaces = mesh.cells[iMap].data

            for iSide in iBCsides:
                for node in mapFaces[iSide]:
                    # Create a unique key for (node, boundary) pair.
                    key_pair = (node, bc_key)

                    # Ignore nodes that have already been processed for this boundary
                    if key_pair in node_bc_translation:
                        continue

                    # Create the new periodic node by applying the boundary's translation.
                    new_node    = mesh.points[node] + VV
                    mesh.points = np.vstack((mesh.points, new_node))
                    node_bc_translation[key_pair] = len(mesh.points) - 1

    # At this point, each (node, bc_key) pair has its own new node
    # > Store these in a mapping (here, keys remain as tuples) for later reference
    periNodes = node_bc_translation.copy()

    # Eliminate duplicate points
    mesh_vars.mesh.points, inverseIndices = np.unique(mesh_vars.mesh.points, axis=0, return_inverse=True)

    # Update the mesh
    for cell in mesh_vars.mesh.cells:
        # Map the old indices to the new ones
        # cell.data = np.vectorize(lambda idx: inverseIndices[idx])(cell.data)
        # Efficiently map all indices in one operation
        cell.data = inverseIndices[cell.data]

    # Update periNodes accordingly
    tmpPeriNodes = {}
    for (node, bc_key), new_node in periNodes.items():
        tmpPeriNodes[(inverseIndices[node], bc_key)] = inverseIndices[new_node]
    periNodes = copy.copy(tmpPeriNodes)
    del tmpPeriNodes

    # Also, remove near duplicate points
    # Create a KDTree for the mesh points
    points = mesh_vars.mesh.points
    tree   = spatial.KDTree(points)

    tol = mesh_vars.tolExternal
    bb  = 0.
    for cell in mesh_vars.mesh.cells:
        # Only consider three-dimensional types
        if not any(s in cell.type for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        # Find the bounding box of the smallest element
        bb = np.min([np.ptp(points[c], axis=0) for c in cell.data])
    # Set the tolerance to 10% of the bounding box of the smallest element
    tol = np.max([tol, bb / ((mesh_vars.nGeo+1)*10.) ])

    # Find all points within the tolerance
    clusters = tree.query_ball_point(points, r=tol)
    del tree

    # Map each point to its cluster representative (first point in the cluster)
    indices = {}
    for i, cluster in enumerate(clusters):
        # Choose the minimum index as the representative for consistency
        representative = min(cluster)
        indices[i] = representative

    # Create a mapping from old indices to new indices
    indices = np.array([indices[i] for i in range(len(points))])

    # Eliminate duplicates
    _, inverseIndices = np.unique(indices, return_inverse=True)
    mesh_vars.mesh.points = points[np.unique(indices)]
    del indices

    # Update the mesh cells
    for cell in mesh_vars.mesh.cells:
        cell.data = inverseIndices[cell.data]

    # Update the periodic nodes
    tmpPeriNodes = {}
    for (node, bc_key), new_node in periNodes.items():
        tmpPeriNodes[(inverseIndices[node], bc_key)] = inverseIndices[new_node]
    mesh_vars.periNodes = tmpPeriNodes

    del inverseIndices

    # Run garbage collector to release memory
    gc.collect()
