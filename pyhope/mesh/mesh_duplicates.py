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
from typing import Final, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.spatial import KDTree
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

    bcs:   Final[list] = mesh_vars.bcs
    vvs:   Final[list] = mesh_vars.vvs

    # Native meshio data
    mesh               = mesh_vars.mesh
    points: np.ndarray = mesh.points
    cells: Final[list] = mesh.cells
    csets: Final[dict] = mesh.cell_sets
    cdict: Final[dict] = mesh.cells_dict

    # Find the mapping to the (N-1)-dim elements
    csetMap = { key: tuple(i for i, cell in enumerate(cset) if cell is not None and cast(np.ndarray, cell).size > 0)
                             for key, cset in csets.items()}

    # Create new periodic nodes per (original node, boundary) pair
    # > Use a dictionary mapping (node, bc_key) --> new node index
    node_bc_translation = {}
    # > Create a list of points to append to the mesh
    pointl  = cast(list, points.tolist())

    for bc_key, cset in csets.items():
        # Find the matching boundary condition
        bcID = find_bc_index(bcs, bc_key)

        # Ignore the volume zones
        volumeBC = False
        for iMap in csetMap[bc_key]:
            if not any(s in tuple(cdict)[iMap] for s in ['quad', 'triangle']):
                volumeBC = True
                break
        if volumeBC:
            continue

        if bcID is None:
            hopout.error(f'Could not find BC {bc_key} in list, exiting...')

        # Only process periodic boundaries in the positive direction
        if bcs[bcID].type[0] != 1 or bcs[bcID].type[3] < 0:
            continue

        iVV = bcs[bcID].type[3]
        VV  = vvs[np.abs(iVV)-1]['Dir'] * np.sign(iVV)

        for iMap in csetMap[bc_key]:
            # Only process 2D faces (quad or triangle)
            if not any(s in tuple(cdict)[iMap] for s in ['quad', 'triangle']):
                continue

            iBCsides = np.array(cset[iMap]).astype(int)
            mapFaces = cells[iMap].data

            for iSide in iBCsides:
                for node in mapFaces[iSide]:
                    # Create a unique key for (node, boundary) pair.
                    key_pair = (node, bc_key)

                    # Ignore nodes that have already been processed for this boundary
                    if key_pair in node_bc_translation:
                        continue

                    # Create the new periodic node by applying the boundary's translation.
                    new_node    = points[node] + VV
                    # mesh.points = np.vstack((mesh.points, new_node))
                    pointl.append(new_node)
                    node_bc_translation[key_pair] = len(pointl) - 1

    # Convert the list of points back to an array
    points = np.array(pointl)
    mesh_vars.mesh.points = points
    del pointl

    # At this point, each (node, bc_key) pair has its own new node
    # > Store these in a mapping (here, keys remain as tuples) for later reference
    periNodes = node_bc_translation.copy()

    # Eliminate duplicate points
    points, inverseIndices = np.unique(points, axis=0, return_inverse=True)

    # Update the mesh
    for cell in cells:
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
    mesh_vars.mesh.points = points
    tree   = KDTree(points)

    # Filter the valid three-dimensional cell types
    valid_cells = tuple(cell for cell in cells if any(s in cell.type for s in mesh_vars.ELEMTYPE.type.keys()))

    tol = mesh_vars.tolExternal
    bbs = np.empty(len(valid_cells), dtype=float)

    for i, cell in enumerate(valid_cells):
        edata   = np.array(cell.data)
        ecoords = points[edata]
        # Compute the ptp (range) along the vertex axis (axis=1) for each element.
        ptp     = np.ptp(ecoords, axis=1)
        # For each element type, take the minimum across dimensions
        bbs[i]  = ptp.min(axis=1).min()

    # Set the tolerance to 10% of the bounding box of the smallest element
    tol = np.max([tol, bbs.min() / ((mesh_vars.nGeo+1)*10.) ])

    # Find all points within the tolerance
    clusters = tree.query_ball_point(points, r=tol)
    del tree

    # Map each point to its cluster representative (first point in the cluster)
    # > Choose the minimum index as the representative for consistency
    indices = np.fromiter((min(cluster) for cluster in clusters), dtype=int)

    # Eliminate duplicates
    _, inverseIndices = np.unique(indices, return_inverse=True)
    mesh_vars.mesh.points = points[np.unique(indices)]
    del indices

    # Update the mesh cells
    for cell in cells:
        cell.data = inverseIndices[cell.data]

    # Update the periodic nodes
    tmpPeriNodes = {}
    for (node, bc_key), new_node in periNodes.items():
        tmpPeriNodes[(inverseIndices[node], bc_key)] = inverseIndices[new_node]
    mesh_vars.periNodes = tmpPeriNodes

    del inverseIndices

    # Run garbage collector to release memory
    gc.collect()
