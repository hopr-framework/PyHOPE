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
    # ------------------------------------------------------
    hopout.routine('Removing duplicate points')

    # Eliminate duplicate points
    mesh_vars.mesh.points, inverseIndices = np.unique(mesh_vars.mesh.points, axis=0, return_inverse=True)

    # Update the mesh
    for cell in mesh_vars.mesh.cells:
        # Map the old indices to the new ones
        # cell.data = np.vectorize(lambda idx: inverseIndices[idx])(cell.data)
        # Efficiently map all indices in one operation
        cell.data = inverseIndices[cell.data]

    # Also, remove near duplicate points
    # Create a KDTree for the mesh points
    points = mesh_vars.mesh.points
    tree   = spatial.KDTree(points)

    # Find all points within the tolerance
    clusters = tree.query_ball_point(points, r=mesh_vars.tolExternal)

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

    # Update the mesh cells
    for cell in mesh_vars.mesh.cells:
        cell.data = inverseIndices[cell.data]
