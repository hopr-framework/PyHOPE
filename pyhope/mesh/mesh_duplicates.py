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
from collections import defaultdict
from typing import Dict, Final, Tuple, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def _unionFind(parent: np.ndarray, x: int) -> int:
    # Path compression
    par = parent  # local ref
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def _unionUnion(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra, rb = _unionFind(parent, a), _unionFind(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def _findPointsTol(points: np.ndarray, tol: float, method: str = 'union_find') -> np.ndarray:
    """ Build an undirected connectivity graph for points within 'tol', then compute
        the connected components and pick the minimum index in each component as the
        representative
    """

    nPoints = points.shape[0]
    match nPoints:
        case 0:  # pragma: no cover
            return np.empty(0, dtype=int)
        case 1:  # pragma: no cover
            return np.zeros(1, dtype=int)

    # Create a KDTree for the mesh points
    tree = KDTree(points)

    match method:
        case 'union_find':
            # Get all unordered pairs (i < j) within tolerance
            # > query_pairs returns a set-like structure; request ndarray for vectorization
            pairs = tree.query_pairs(tol, output_type='ndarray')

            if pairs.size == 0:  # pragma: no cover
                # All isolated: each point is its own representative
                return np.arange(nPoints, dtype=int)

            # Disjoint Set (Union-Find)
            parent = np.arange(nPoints, dtype=int)
            rank   = np.zeros(nPoints, dtype=int)

            # Union all pairs
            for a, b in pairs:
                _unionUnion(parent, rank, int(a), int(b))

            # Final pass: compress and compute representatives (minimum index per root)
            # > Find root for every point
            for i in range(nPoints):
                parent[i] = _unionFind(parent, i)
            components, labels = nPoints, parent

        case 'sparse':  # pragma: no cover
            # Construct a sparse adjacency matrix where edges connect points within 'tol'
            # > Use sparse_distance_matrix to avoid Python-level loops and return COO/CSR in C
            #
            # NOTE: This includes self-distances (diagonal); zero them below
            adj = tree.sparse_distance_matrix(tree, tol, output_type='coo_matrix').tocsr()

            # Remove self-connections, enforce symmetry
            adj.setdiag(0)
            adj.eliminate_zeros()
            # Make matrix symmetric (in case of any asymmetry)
            adj = adj.maximum(adj.T)

            # Ensure canonical CSR for faster graph ops
            adj.sum_duplicates()
            adj.sort_indices()

            # If there are no edges (all points isolated w.r.t. tol), each point is its own component
            nPoints = points.shape[0]
            if adj.nnz == 0:  # pragma: no cover
                return np.arange(nPoints, dtype=int)

            # Compute connected components (undirected)
            components, labels = connected_components(adj, directed=False, return_labels=True)

        case _:  # pragma: no cover
            raise ValueError('Unknown method in _findPointsTol')

    # For each component label, choose the minimum original point index as representative
    repLabel  = np.full(components, nPoints, dtype=int)
    # Assign each point its component representative
    np.minimum.at(repLabel, labels, np.arange(nPoints, dtype=int))
    repsPoint = repLabel[labels]

    return repsPoint


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
    csetMap: Dict      = { key: tuple(i for i, cell in enumerate(cset) if cell is not None and cast(np.ndarray, cell).size > 0)
                                        for key, cset in csets.items()}

    # Create new periodic nodes per (original node, boundary) pair
    # > Use a dictionary mapping (node, bc_key) --> new node index
    nodeTrans: Dict[Tuple[int, str], int] = {}
    # > Collect points to append to the mesh
    newPoints: list       = []
    nPoints:   Final[int] = points.shape[0]
    BCNodes:   Dict       = {}

    for bc_key, cset in csets.items():
        # Find the matching boundary condition
        bcID = find_bc_index(bcs, bc_key)

        # Ignore the volume zones
        if any(not any(s in tuple(cdict)[iMap] for s    in ('quad', 'triangle'))  # noqa: E272
                                               for iMap in csetMap[bc_key]):
            continue

        # Error if BC has no ID
        if bcID is None:
            hopout.error(f'Could not find BC {bc_key} in list, exiting...')

        # Only process periodic boundaries in the positive direction
        if bcs[bcID].type[0] != 1 or bcs[bcID].type[3] < 0:
            continue

        iVV = bcs[bcID].type[3]
        VV  = vvs[np.abs(iVV)-1]['Dir'] * np.sign(iVV)

        currentBCNodes = set()
        for iMap in csetMap[bc_key]:
            # Only process 2D faces (quad or triangle)
            if any(s in tuple(cdict)[iMap] for s in ('quad', 'triangle')):
                mapFaces = cells[iMap].data
                currentBCNodes.update(node for iSide in cset[iMap] for node in mapFaces[iSide])

        # Ignore nodes that have already been processed for this boundary
        if bc_key not in BCNodes:
            BCNodes[bc_key] = set()

        currentNodes = list(currentBCNodes - BCNodes[bc_key])
        BCNodes[bc_key].update(currentNodes)

        if not currentNodes:
            continue

        # Create the new periodic node by applying the boundary's translation
        newNodes = points[currentNodes] + VV
        newPoints.extend(newNodes)

        # Update translation dictionary
        start_index = nPoints + len(newPoints) - len(currentNodes)
        for i, node in enumerate(currentNodes):
            nodeTrans[(node, bc_key)] = start_index + i

    # Append new periodic nodes (if any) to the mesh
    if newPoints:
        points = np.vstack((points, np.asarray(newPoints)))
    del newPoints

    # At this point, each (node, bc_key) pair has its own new node
    # > Store these in a mapping (here, keys remain as tuples) for later reference
    periNodes = nodeTrans.copy()

    # Eliminate duplicate points
    points, inverseIndices = np.unique(points, axis=0, return_inverse=True)
    # PERF: This should be faster but produces slightly wrong results
    # # > Create a 1D view of the 2D points array where each row is a single item
    # voidView = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * points.shape[1])))
    # # > Use np.unique on the 1D view
    # _, uniqueIndices, inverseIndices = np.unique(voidView, return_index=True, return_inverse=True)
    # # > Reconstruct the unique points array from the original points using the unique_indices
    # points, inverseIndices = points[uniqueIndices], inverseIndices.reshape(-1)
    # del voidView, uniqueIndices

    # Update the mesh
    for cell in cells:
        # Map the old indices to the new ones
        cell.data = inverseIndices[cell.data]

    # Update periNodes accordingly
    periNodes = { (inverseIndices[node], bc_key): inverseIndices[new_node] for (node, bc_key), new_node in periNodes.items() }

    # Also, remove near duplicate points
    # > Filter the valid three-dimensional cell types
    valid_cells = tuple(cell for cell in cells if any(s in cell.type for s in mesh_vars.ELEMTYPE.type.keys()))
    # > Group by number of vertices per element to avoid ragged arrays
    groups = defaultdict(list)
    for cell in valid_cells:
        groups[cell.data.shape[1]].append(cell.data)

    bbs = float('inf')
    for blocks in groups.values():
        # Concatenate all elements with the same vertex count
        cell_data = np.concatenate(blocks, axis=0)
        coords    = points[cell_data]

        # Compute the ptp (range) along the vertex axis (axis=1) for each element
        ptp = np.ptp(coords, axis=1)
        # For each element type, take the minimum across dimensions
        bbs = min(bbs, ptp.min())

    # Set the tolerance to 10% of the bounding box of the smallest element
    tol = np.max([mesh_vars.tolExternal, bbs / ((mesh_vars.nGeo+1)*10.) if bbs != float('inf') else 0.0])

    # Find all points within the tolerance
    reps = _findPointsTol(points, tol, method='union_find')

    # Eliminate duplicates
    # > reps[i] is the chosen representative index for point i
    indices, inverseIndices = np.unique(reps, return_inverse=True)
    mesh_vars.mesh.points = points[indices]
    del reps, indices

    # Update the mesh cells
    for cell in cells:
        cell.data = inverseIndices[cell.data]

    # Update the periodic nodes
    periNodes = { (inverseIndices[node], bc_key): inverseIndices[new_node] for (node, bc_key), new_node in periNodes.items() }
    mesh_vars.periNodes = periNodes

    del inverseIndices

    # Run garbage collector to release memory
    gc.collect()
