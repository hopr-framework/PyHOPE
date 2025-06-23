#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (C) 2022 Nico Schlömer
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
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def gmsh_to_meshio(gmsh) -> meshio.Mesh:
    """
    Convert a Gmsh object to a meshio object.
    """
    # Local imports ----------------------------------------
    from pyhope.meshio.meshio_ordering import NodeOrdering
    # ------------------------------------------------------

    # Initialize the node ordering
    node_ordering = NodeOrdering()

    # Extract point coords
    idx, points, _ = gmsh.model.mesh.getNodes()
    points  = np.asarray(points).reshape(-1, 3)
    idx    -= 1
    srt     = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    points = points[srt]

    # Extract cells
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements()
    cells = []
    for elem_type, elem_tags, node_tags in zip(elem_types, elem_tags, node_tags):
        # `elementName', `dim', `order', `numNodes', `localNodeCoord', `numPrimaryNodes'
        num_nodes_per_cell = gmsh.model.mesh.getElementProperties(elem_type)[3]

        node_tags_reshaped = np.asarray(node_tags).reshape(-1, num_nodes_per_cell) - 1
        node_tags_reshaped = node_ordering.ordering_gmsh_to_meshio(elem_type, node_tags_reshaped)

        # NRG: Fix the element ordering
        node_tags_sorted   = node_tags_reshaped[np.argsort(elem_tags)]
        cells.append(meshio.CellBlock(meshio.gmsh.gmsh_to_meshio_type[elem_type], node_tags_sorted))

    cell_sets = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        # Get offset of the node tags (gmsh sorts elements of all dims in succeeding order of node tags, but order of dims might differ)
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=dim)
        offset = min(elem_tags[0])

        name = gmsh.model.getPhysicalName(dim, tag)
        cell_sets[name] = [[] for _ in range(len(cells))]
        for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            # elem_types, elem_tags, node_tags
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, e)
            assert len(elem_types) == len(elem_tags)
            assert len(elem_types) == 1
            elem_type = elem_types[0]
            elem_tags = elem_tags[0] - offset

            meshio_cell_type = meshio.gmsh.gmsh_to_meshio_type[elem_type]
            # Make sure that the cell type appears only once in the cell list
            idx = []
            for k, cell_block in enumerate(cells):
                if cell_block.type == meshio_cell_type:
                    idx.append(k)
            assert len(idx) == 1
            idx = idx[0]
            cell_sets[name][idx].append(elem_tags)

        cell_sets[name] = [(None if len(idcs) == 0 else np.concatenate(idcs)) for idcs in cell_sets[name]]

    return meshio.Mesh(points, cells, cell_sets=cell_sets)
