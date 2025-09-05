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
from collections import defaultdict
from typing import Dict, Final, List, Set, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
import h5py
import numpy as np
from meshio import CellBlock
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def DefineIO() -> None:
    # Local imports ----------------------------------------
    from pyhope.io.io_vars import MeshFormat
    from pyhope.readintools.readintools import CreateIntFromString, CreateIntOption, CreateLogical, CreateSection, CreateStr
    # ------------------------------------------------------

    CreateSection('Output')
    CreateStr('ProjectName', help='Name of output files')
    CreateIntFromString('OutputFormat'  , default='HDF5', help=f'Mesh output format [{", ".join(s.name for s in MeshFormat)}]')
    CreateIntOption(    'OutputFormat'  , number=MeshFormat.HDF5.value, name=MeshFormat.HDF5.name)
    CreateIntOption(    'OutputFormat'  , number=MeshFormat.VTK.value , name=MeshFormat.VTK.name)
    CreateIntOption(    'OutputFormat'  , number=MeshFormat.GMSH.value, name=MeshFormat.GMSH.name)
    CreateLogical(      'DebugMesh'     , default=False , help='Output debug mesh in XDMF format')
    CreateLogical(      'DebugVisu'     , default=False , help='Launch the GMSH GUI to visualize the mesh')


def InitIO() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import GetIntFromStr, GetLogical, GetStr
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('INIT OUTPUT...')

    io_vars.projectname  = GetStr('ProjectName')
    io_vars.outputformat = GetIntFromStr('OutputFormat')

    io_vars.debugmesh    = GetLogical('DebugMesh')
    io_vars.debugvisu    = GetLogical('DebugVisu')

    # hopout.info('INIT OUTPUT DONE!')


def IO() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    from pyhope.mesh.mesh_common import edges
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Common
    from pyhope.io.io_debug import DebugIO
    from pyhope.io.io_gmsh import GMSHCELLTYPES
    from pyhope.io.io_vars import MeshFormat, ELEM, ELEMTYPE
    from pyhope.meshio.meshio_nodes import NumNodesPerCell
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('OUTPUT MESH...')

    pname:  Final[str] = io_vars.projectname

    match io_vars.outputformat:
        case MeshFormat.HDF5.value:
            mesh  = mesh_vars.mesh
            elems: Final[list] = cast(list, mesh_vars.elems)
            sides: Final[list] = cast(list, mesh_vars.sides)
            bcs:   Final[list] = cast(list, mesh_vars.bcs)

            nElems: Final[int] = len(elems)
            nSides: Final[int] = len(sides)
            nBCs:   Final[int] = len(bcs)
            # Number of non-unique nodes, vertices, edges
            nNodes:    Final[int] = np.array([elem.nodes.size       for elem in elems], dtype=np.int32).sum(dtype=int)  # noqa: E272
            nVertices: Final[int] = np.array([elem.type % 10        for elem in elems], dtype=np.int32).sum(dtype=int)  # noqa: E272
            nEdges:    Final[int] = np.array([len(edges(elem.type)) for elem in elems], dtype=np.int32).sum(dtype=int)  # noqa: E272

            fname = '{}_mesh.h5'.format(pname)

            elemInfo, elemIJK, sideInfo, nodeInfo, nodeCoords, \
            FEMElemInfo, nFEMVertices, vertexInfo, vertexConnectInfo, nFEMEdges, edgeInfo, edgeConnectInfo, \
            elemCounter = getMeshInfo()

            # Print the final output
            hopout.sep()
            elem_types = [(elemType, count) for elemType in ELEM.TYPES if (count := elemCounter.get(elemType, 0)) > 0]
            for elemType, count in elem_types:
                hopout.info(f'{ELEMTYPE(elemType)}: {count:12d}')

            hopout.sep()
            hopout.routine('Writing HDF5 mesh to "{}"'.format(fname))

            with h5py.File(fname, mode='w') as f:
                # Store same basic information
                common = Common()
                f.attrs['HoprVersion'   ] = common.version
                f.attrs['HoprVersionInt'] = common.__version__.micro + common.__version__.minor*100 + common.__version__.major*10000

                # Store mesh information
                f.attrs['Ngeo'          ] = mesh_vars.nGeo
                f.attrs['nElems'        ] = nElems
                f.attrs['nSides'        ] = nSides
                f.attrs['nNodes'        ] = nNodes
                f.attrs['nVertices'     ] = nVertices
                f.attrs['nEdges'        ] = nEdges
                f.attrs['nUniqueSides'  ] = np.max(sideInfo[:, 1])
                f.attrs['nUniqueNodes'  ] = np.max(nodeInfo)

                _ = f.create_dataset('ElemInfo'     , data=elemInfo)
                _ = f.create_dataset('ElemCounter'  , data=np.array(list(elemCounter.items()), dtype=np.int32))
                _ = f.create_dataset('SideInfo'     , data=sideInfo)
                _ = f.create_dataset('GlobalNodeIDs', data=nodeInfo)
                _ = f.create_dataset('NodeCoords'   , data=nodeCoords)

                if elemIJK     is not None:  # noqa: E272
                    _ = f.create_dataset('nElems_IJK'        , data=mesh_vars.nElemsIJK)
                    _ = f.create_dataset('Elem_IJK'          , data=elemIJK)

                if FEMElemInfo is not None:
                    # Store FEM information
                    f.attrs['FEMconnect'] = 'ON'
                    f.attrs['nFEMVertices'         ] = nFEMVertices
                    f.attrs['nFEMVertexConnections'] = vertexConnectInfo.shape[0]
                    f.attrs['nFEMEdges'            ] = nFEMEdges
                    f.attrs['nFEMEdgeConnections'  ] = edgeConnectInfo  .shape[0]
                    # TODO: This seems to be just repeated information
                    f.attrs['nUniqueEdges'         ] = nFEMEdges

                    _ = f.create_dataset('FEMElemInfo'       , data=FEMElemInfo)
                    _ = f.create_dataset('VertexInfo'        , data=vertexInfo)
                    _ = f.create_dataset('VertexConnectInfo' , data=vertexConnectInfo)
                    _ = f.create_dataset('EdgeInfo'          , data=edgeInfo)
                    _ = f.create_dataset('EdgeConnectInfo'   , data=edgeConnectInfo)

                # Store boundary information
                f.attrs['nBCs'          ] = nBCs
                bcNames = [f'{bc.name:<255}' for bc in bcs]
                bcTypes = np.array([bc.type  for bc in bcs], dtype=np.int32).reshape(-1, 4)  # noqa: E272

                _ = f.create_dataset('BCNames'   , data=np.array(bcNames, dtype='S'))
                _ = f.create_dataset('BCType'    , data=bcTypes)

                # Check if there is a periodic vector and write it to mesh file
                nVV = len(mesh_vars.vvs)
                if nVV > 0:
                    vvs = np.array([[vv['Dir'][i] for i in range(3)] for vv in mesh_vars.vvs], dtype=np.float64)
                    _ = f.create_dataset('VV', data=vvs)

                # Write a low-order debug mesh if requested
                if io_vars.debugmesh:
                    DebugIO()

        case MeshFormat.VTK.value:
            mesh  = mesh_vars.mesh
            fname = '{}_mesh.vtk'.format(pname)

            hopout.sep()
            hopout.routine('Writing VTK mesh to "{}"'.format(fname))

            mesh.write(fname, file_format='vtk42')

        case MeshFormat.GMSH.value:
            if mesh_vars.nGeo > 1:
                hopout.sep()
                print(hopout.warn('GMSH output is incomplete for nGeo > 1'))

            mesh  = mesh_vars.mesh
            fname = '{}_mesh.msh'.format(pname)

            # Instantiate the Gmsh cell type mapping
            gmshCellTypes = GMSHCELLTYPES()

            # Combine volume and surface cells as separate cell blocks
            volume_cells  = [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes3D]
            surface_cells = [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D]

            # Print the final output
            hopout.sep()
            numNodes = NumNodesPerCell()
            for cell in volume_cells:
                cellType  = ''.join([s for s in cell.type if not s.isdigit()])
                cellNodes = numNodes[cellType]
                elemOrder = 100 if not any(s.isdigit() for s in cell.type) else 200
                elemType  = cellNodes + elemOrder
                hopout.info(f'{ELEMTYPE(elemType)}: {len(cell):12d}')

            # Build new arrays
            celll:      List[CellBlock ] = []
            celldphys:  List[np.ndarray] = []
            celldgeom:  List[np.ndarray] = []

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
                cb     = CellBlock(cell_block.type, cell_block.data.copy())
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

                    cb     = CellBlock(cell_type, cell_block.data[sel].copy())
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
            gmshMesh.cell_data.update({'gmsh:physical'   : celldphys})   # pyright: ignore[reportArgumentType, reportCallIssue]
            gmshMesh.cell_data.update({'gmsh:geometrical': celldgeom})   # pyright: ignore[reportArgumentType, reportCallIssue]

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

            hopout.sep()
            hopout.routine('Writing GMSH mesh to "{}"'.format(fname))
            gmshMesh.write(fname, file_format='gmsh')

        case _:  # Default
            hopout.error('Unknown output format {}, exiting...'.format(io_vars.outputformat))

    # hopout.sep()
    # hopout.info('OUTPUT MESH DONE!')


def getMeshInfo() -> tuple[np.ndarray,         # ElemInfo
                           np.ndarray | None,  # ElemIJK
                           np.ndarray,         # SideInfo
                           np.ndarray,         # NodeInfo
                           np.ndarray,         # NodeCoords
                           np.ndarray | None,  # Optional[FEMElemInfo]
                           int        | None,  # Optional[nVertices]
                           np.ndarray | None,  # Optional[VertexInfo]
                           np.ndarray | None,  # Optional[VertexConnectInfo]
                           int        | None,  # Optional[nEdges]
                           np.ndarray | None,  # Optional[EdgeInfo]
                           np.ndarray | None,  # Optional[EdgeConnectInfo]
                           dict[int, int]
                          ]:
    # Standard libraries -----------------------------------
    import heapq
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.fem.fem import getFEMInfo
    from pyhope.mesh.mesh_common import LINTEN
    from pyhope.io.io_vars import ELEM, SIDE
    # ------------------------------------------------------

    mesh:   Final             = mesh_vars.mesh
    elems:  Final[list]       = mesh_vars.elems
    sides:  Final[list]       = mesh_vars.sides
    points: Final[np.ndarray] = mesh.points

    nElems: Final[int] = len(elems)
    nSides: Final[int] = len(sides)

    # Create the ElemCounter
    elemCounter = defaultdict(int)
    for elemType in ELEM.TYPES:
        elemCounter[elemType] = 0

    # Pre-allocate arrays
    elemInfo  = np.zeros((nElems, ELEM.INFOSIZE), dtype=np.int32)
    # sideCount = 0  # elem['Sides'] might work as well
    # nodeCount = 0  # elem['Nodes'] contains the unique nodes

    # Calculate the ElemInfo
    elem_types = np.array([elem.type                                 for elem in elems], dtype=np.int32)  # noqa: E272
    elem_zones = np.array([elem.zone if elem.zone is not None else 1 for elem in elems], dtype=np.int32)  # noqa: E272
    elem_sides = np.array([len(elem.sides)                           for elem in elems], dtype=np.int32)  # noqa: E272
    elem_nodes = np.array([elem.nodes.size                           for elem in elems], dtype=np.int32)  # noqa: E272

    # Fill basic element info
    elemInfo[:, ELEM.TYPE] = elem_types
    elemInfo[:, ELEM.ZONE] = elem_zones

    # Calculate cumulative sums
    side_cumsum = np.concatenate([[0], np.cumsum(elem_sides)])
    node_cumsum = np.concatenate([[0], np.cumsum(elem_nodes)])

    # Fill element side info
    elemInfo[:, ELEM.FIRSTSIDE] = side_cumsum[ :-1]
    elemInfo[:, ELEM.LASTSIDE ] = side_cumsum[1:]
    elemInfo[:, ELEM.FIRSTNODE] = node_cumsum[ :-1]
    elemInfo[:, ELEM.LASTNODE ] = node_cumsum[1:]

    # Update element counter
    uniq_types, uniq_counts = np.unique(elem_types, return_counts=True)
    for elemType, elemCount in zip(uniq_types, uniq_counts):
        elemCounter[elemType] = elemCount

    # Fill the IJK-sorting array
    elemIJK = None
    if hasattr(mesh_vars, 'nElemsIJK'):
        elemIJK = np.vstack([elem.elemIJK for elem in elems]).astype(np.int32)

    # Set the global side ID
    globalSideID     = 0
    highestSideID    = 0
    usedSideIDs      = set()  # Set to track used side IDs
    availableSideIDs = []     # Min-heap for gap

    for side in sides:
        # Already counted the side
        if side.globalSideID is not None:
            continue

        # Get the smallest available globalSideID from the heap, if any
        if availableSideIDs:
            globalSideID = heapq.heappop(availableSideIDs)
        else:
            # Use the current maximum ID and increment
            globalSideID = highestSideID + 1

        # Mark the side ID as used
        highestSideID = max(globalSideID, highestSideID)
        usedSideIDs.add(globalSideID)
        # side.update(globalSideID=globalSideID)
        side.globalSideID = globalSideID

        if side.connection is None or side.connection < 0:  # BC/big mortar side
            pass
        elif side.MS == 1:                                  # Internal / periodic side (master side)
            # Get the connected slave side
            nbSideID = side.connection

            # Reclaim the ID of the slave side if already assigned
            if sides[nbSideID].globalSideID is not None:
                reclaimedID = sides[nbSideID].globalSideID
                usedSideIDs.remove(reclaimedID)
                heapq.heappush(availableSideIDs, reclaimedID)

            # Set the negative globalSideID of the slave side
            # sides[nbSideID].update(globalSideID=-(globalSideID))
            sides[nbSideID].globalSideID = -(globalSideID)

    # If there are any gaps in the side IDs, fill them by reassigning consecutive values
    if availableSideIDs:
        # Collect all master sides (globalSideID > 0) and sort them by their current IDs
        masters = sorted((side for side in sides if side.globalSideID > 0), key=lambda side: side.globalSideID)

        # Build a mapping from old master ID to new consecutive IDs (starting at 1)
        mapping = {side.globalSideID: newID for newID, side in enumerate(masters, start=1)}

        # Update the sides based on the mapping
        for side in sides:
            # For slave sides, update to the negative of the mapped master ID
            side.globalSideID = mapping[side.globalSideID] if side.globalSideID > 0 else -mapping[-side.globalSideID]

    # Pre-allocate arrays
    sideInfo   = np.zeros((nSides, SIDE.INFOSIZE), dtype=np.int32)

    # Calculate the SideInfo
    side_types = np.array([side.sideType     for side in sides], dtype=np.int32)  # noqa: E272
    side_gloID = np.array([side.globalSideID for side in sides], dtype=np.int32)  # noqa: E272

    # Fill basic side info
    sideInfo[:, SIDE.TYPE] = side_types
    sideInfo[:, SIDE.ID  ] = side_gloID

    # Process side connections
    for iSide, side in enumerate(sides):
        # Connected sides
        if side.connection is None:                                # BC side
            sideInfo[iSide, SIDE.NBELEMID      ] = 0
            sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = 0
            sideInfo[iSide, SIDE.BCID          ] = side.bcid + 1
        elif side.locMortar is not None:                           # Small mortar side
            nbSideID = side.connection
            nbElemID = sides[nbSideID].elemID + 1  # Python -> HOPR index
            sideInfo[iSide, SIDE.NBELEMID      ] = nbElemID
        elif side.connection is not None and side.connection < 0:  # Big mortar side
            # WARNING: This is not a sideID, but the mortar type
            sideInfo[iSide, SIDE.NBELEMID      ] = side.connection
            # Periodic mortar sisters have a BCID
            if side.bcid is not None:
                sideInfo[iSide, SIDE.BCID      ] = side.bcid + 1
        else:                                                      # Internal side
            nbSideID = side.connection
            nbElemID = sides[nbSideID].elemID + 1  # Python -> HOPR index
            sideInfo[iSide, SIDE.NBELEMID      ] = nbElemID
            if side.sideType < 0:    # Small mortar side
                sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = side.flip
            elif side.flip == 0:     # Master side
                sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = sides[nbSideID].locSide*10
            else:
                sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = sides[nbSideID].locSide*10 + side.flip

            # Periodic/inner sides still have a BCID
            if side.bcid is not None:
                sideInfo[iSide, SIDE.BCID      ] = side.bcid + 1
            else:
                sideInfo[iSide, SIDE.BCID      ] = 0

    # Pre-allocate arrays
    nNodes: Final[int] = elem_nodes.sum(dtype=int)  # number of non-unique nodes
    nodeInfo   = np.zeros((nNodes)   , dtype=np.int32)
    nodeCoords = np.zeros((nNodes, 3), dtype=np.float64)

    # Pre-compute LINTEN mappings for all element types
    elemTypes = np.unique(elemInfo[:, 0])
    linCache  = {}
    for elemType in elemTypes:
        _, mapLin = LINTEN(elemType, order=mesh_vars.nGeo)
        mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))
        linCache[elemType] = mapLin

    # Calculate the NodeInfo
    nodeCount = 0
    for elem in elems:
        # Mesh coordinates are stored in meshIO sorting
        elemType = elem.type
        mapLin   = linCache[elemType]

        elemNodes  = np.asarray(elem.nodes)
        nElemNodes = elemNodes.size
        indices    = nodeCount + mapLin[:nElemNodes]

        # Assign nodeInfo and nodeCoords in vectorized fashion
        nodeInfo[  indices] = elemNodes + 1
        nodeCoords[indices] = points[elemNodes]

        nodeCount += nElemNodes

    if hasattr(elems[0], 'vertexInfo') and elems[0].vertexInfo is not None:
        FEMElemInfo, nFEMVertices, vertexInfo, vertexConnectInfo, nFEMEdges, edgeInfo, edgeConnectInfo = getFEMInfo(nodeInfo)
    else:
        nFEMVertices = nFEMEdges  = 0
        FEMElemInfo  = vertexInfo = vertexConnectInfo = edgeInfo = edgeConnectInfo = None

    return elemInfo, elemIJK, sideInfo, nodeInfo, nodeCoords, \
           FEMElemInfo, nFEMVertices, vertexInfo, vertexConnectInfo, nFEMEdges, edgeInfo, edgeConnectInfo, \
           elemCounter
