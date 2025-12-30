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
import importlib
from typing import Dict, Final, List, Set, cast
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
        # Get offset of the node tags (gmsh sorts elements of all dims in succeeding order of node tags, but order of dims might differ)  # noqa: E501
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=dim)
        elem_tags_group = {meshio.gmsh.gmsh_to_meshio_type[j]: i for i, j in zip(elem_tags, elem_types)}

        name = gmsh.model.getPhysicalName(dim, tag)
        cell_sets[name] = [[] for _ in range(len(cells))]
        for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            # elem_types, elem_tags, node_tags
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, e)
            assert len(elem_types) == len(elem_tags)

            meshio_cell_type = [meshio.gmsh.gmsh_to_meshio_type[type_ele] for type_ele in elem_types]
            # Make sure that the cell type appears only once in the cell list
            idx = []
            for k, cell_block in enumerate(cells):
                if cell_block.type in meshio_cell_type:
                    idx.append(k)

            offset = {meshio_cell_type[j]: np.where(elem_tags_group[meshio_cell_type[j]] == elem_tags[j][0])[0] for j in range(len(idx))}  # noqa: E501
            elem_tags = [offset[j] + np.int64(i - i[0]) for j, i in zip(meshio_cell_type, elem_tags)]

            for j, i in enumerate(idx):
                cell_sets[name][i].append(elem_tags[j])

        cell_sets[name] = [(None if len(idcs) == 0 else np.concatenate(idcs)) for idcs in cell_sets[name]]

    return meshio.Mesh(points, cells, cell_sets=cell_sets)


def meshio_to_gmsh(mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.io.io_gmsh import GMSHCELLTYPES
    # ------------------------------------------------------

    # Instantiate the Gmsh cell type mapping
    gmshCellTypes = GMSHCELLTYPES()

    # Combine volume and surface cells as separate cell blocks
    volume_cells  = [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes3D]
    surface_cells = [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D]

    # Build new arrays
    celll:      List[meshio.CellBlock] = []
    celldphys:  List[np.ndarray      ] = []
    celldgeom:  List[np.ndarray      ] = []

    # Unique geometrical entity ids per dimension
    # > 0: 3D entities, 1: 2D entities
    geom_id:    np.ndarray       = np.ones((2,), dtype=int)
    geom_tag:   List[List[int]]  = [[] for _ in range(2)]
    geom_nodes: List[List[int]]  = [[] for _ in range(2)]

    # Set to keep track of nodes already used to represent an entity in gmsh:dim_tags
    usedNodes: Set[int] = set()

    # Process each 3D CellBlock: keep shared connectivity, set tags
    # WARNING: Each 3D CellBlock neets to get its OWN geometrical tag so that
    #          meshio._write_entities() does not see duplicate (dim, tag) across cell blocks
    for cell_block in volume_cells:
        # TODO: Get the physical region from mesh_vars.elems
        zone = 1

        # Shared nodes, so keep the original connectivity
        cb     = meshio.CellBlock(cell_block.type, cell_block.data.copy())
        nCells = len(cell_block.data)
        # Unique 3D geometrical tag for volume block
        tag    = int(geom_id[0])
        phys   = np.full(nCells, int(zone), dtype=int)
        geom   = np.full(nCells, tag,       dtype=int)

        # Choose a representative node for this 3D entity (unique across all entities)
        node_cand = cast(np.ndarray, cb.data).ravel().astype(int)
        node_free = [s for s in node_cand if s not in usedNodes]
        node_used = int(node_free[0] if node_free else node_cand[0])
        if not node_free and node_used in usedNodes:
            # All candidate nodes already used; fall back to the first (still valid, we just can't make it unique)
            # hopout.routine(f'Note: reusing representative node {rep_node} for 3D entity tag {tag}')
            hopout.error('All candidate nodes already used for 3D entity tag {tag}')

        geom_nodes[0].append(node_used)
        usedNodes.add(node_used)

        # Update the lists and next 3D entity ID
        geom_id[ 0] += 1
        geom_tag[0].append(tag)
        celll      .append(cb)
        celldphys  .append(phys)
        celldgeom  .append(geom)

    # Build a reverse mapping from the cell sets to the cell types
    all_cells:      Final[tuple] = tuple(cell_block for cell_block in mesh.cells)
    cellTypeToBC:   Final[dict]  = {s.type: {} for s in all_cells}
    cellTypeToName: Final[list]  = [s.type     for s in all_cells]  # noqa: E272
    BCNameToBCID:   Final[dict]  = {bc.name: bc.bcid for bc in mesh_vars.bcs if bc.bcid is not None}
    del all_cells

    # Find the mapping from surface to BC for all cell sets
    for setID, cellSet in mesh.cell_sets.items():
        for cellID, setCells in enumerate(cellSet):
            if setCells is None:
                continue

            cellName = cellTypeToName[cellID]

            for cell_idx in cast(np.ndarray, setCells):
                BCID = BCNameToBCID.get(setID)
                cellTypeToBC[cellName].update({int(cell_idx): BCID})

    # Process each 2D CellBlock: keep shared point indices, set BCs
    # NOTE: Split by BC id so each physical surface becomes its own entity
    for _, cell_block in enumerate(surface_cells):
        cell_type = cell_block.type
        # NOTE: This will assign 0 to faces not found, assuming they are internal faces
        cellBC = np.asarray([cellTypeToBC[cell_type].get(int(idx), 0) for idx in range(len(cell_block))], dtype=int)

        # Create a separate CellBlock per unique BC id
        for BCID in np.unique(cellBC):
            sel = np.where(cellBC == int(BCID))[0]
            if sel.size == 0:
                continue

            cb     = meshio.CellBlock(cell_type, cell_block.data[sel].copy())
            nCells = len(cell_block.data[sel])
            # Unique 2D geometrical tag for surface block
            tag    = int(geom_id[1])
            phys   = np.full(nCells, int(BCID), dtype=int)
            geom   = np.full(nCells, tag      , dtype=int)

            # Choose a representative node for this 2D entity (unique across all entities)
            node_cand = cast(np.ndarray, cb.data).ravel().astype(int)
            node_free = [s for s in node_cand if s not in usedNodes]
            node_used = int(node_free[0] if node_free else node_cand[0])
            if not node_free and node_used in usedNodes:
                # All candidate nodes already used; fall back to the first (still valid, we just can't make it unique)
                # hopout.routine(f'Note: reusing representative node {rep_node} for 3D entity tag {tag}')
                hopout.error('All candidate nodes already used for 3D entity tag {tag}')

            geom_nodes[1].append(node_used)
            usedNodes.add(node_used)

            # Update lists and next 2D entity ID
            geom_id[ 1] += 1
            geom_tag[1].append(tag)
            celll    .append(cb)
            celldphys.append(phys)
            celldgeom.append(geom)

    # Create new mesh with separate CellBlocks
    gmshMesh = type(mesh)(points = mesh.points,        # noqa: E251
                          cells  = cast(dict, celll))  # noqa: E251

    # Mixed elements require gmsh:physical and gmsh:geometrical
    gmshMesh.cell_data.update({'gmsh:physical'   : cast(list, celldphys)})   # pyright: ignore[reportArgumentType, reportCallIssue]
    gmshMesh.cell_data.update({'gmsh:geometrical': cast(list, celldgeom)})   # pyright: ignore[reportArgumentType, reportCallIssue]

    # Provide entity information for nodes (gmsh:dim_tags)
    # Strategy:
    #  - Default all nodes to the FIRST 3D entity
    #  - Override one unique node per 3D entity with (3, tag) to ensure all 3D entities exist in $Entities
    #  - Override one unique node per 2D entity with (2, tag) to ensure all 2D entities exist in $Entities
    dim_tags = np.zeros((mesh.points.shape[0], 2), dtype=int)
    dim_tags[:, 0] = 3
    dim_tags[:, 1] = int(geom_tag[0][0])

    # Set representatives for ALL 3D entities
    for tag, node_used in zip(geom_tag[0], geom_nodes[0]):
        dim_tags[int(node_used), 0] = 3
        dim_tags[int(node_used), 1] = int(tag)

    # Set representatives for ALL 2D entities
    for tag, node_used in zip(geom_tag[1], geom_nodes[1]):
        dim_tags[int(node_used), 0] = 2
        dim_tags[int(node_used), 1] = int(tag)

    gmshMesh.point_data.update({'gmsh:dim_tags': dim_tags})

    # Add PhysicalNames so groups are not missing in the Gmsh output
    field_data: Dict[str, np.ndarray] = {}

    # Add volume physical group (3D)
    if len(geom_tag[0]) > 0:
        field_data['volume'] = np.array([1, 3], dtype=int)  # id=1, dim=3

    # Add surface physical groups (2D) from boundary conditions
    field_data.update({ name: np.array([int(BCID), 2], dtype=int) for name, BCID in BCNameToBCID.items()
                                                                         if BCID is not None })

    # Update or create field_data attribute
    gmshMesh.field_data = {**getattr(gmshMesh, 'field_data', {}), **field_data}

    return gmshMesh


def MeshioGmshOrderingPatch() -> None:
    """ Monkey-patch MeshIO's Gmsh writer to call NodeOrdering helper
    """

    # Avoid multiple patching
    import meshio.gmsh.common as common
    if getattr(common, '_pyhope_ordering_patched', False):
        return None

    # Lazy import to avoid circulars
    try:
        from pyhope.meshio.meshio_ordering import NodeOrdering
    except Exception:
        return None

    # Patch the common module
    try:
        common._meshio_to_gmsh_order = NodeOrdering().ordering_meshio_to_gmsh
        common._pyhope_ordering_patched = True  # type: ignore[attr-defined]
    except Exception:
        # If assignment fails, bail out
        return

    # Also patch known writer modules that imported the symbol directly at module scope
    # > They reference this wrapper at call time
    for mod_name in ('meshio.gmsh._gmsh', 'meshio.gmsh._gmsh41', 'meshio.gmsh._gmsh22', 'meshio.gmsh._gmsh4'):
        try:
            mod = importlib.import_module(mod_name)
            setattr(mod, '_meshio_to_gmsh_order', NodeOrdering().ordering_meshio_to_gmsh)
        except Exception:
            # If assignment fails, pass
            pass
