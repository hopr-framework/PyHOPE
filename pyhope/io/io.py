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
from collections import OrderedDict
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
import h5py
import numpy as np
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
    CreateIntFromString('OutputFormat', default='HDF5', help='Mesh output format')
    CreateIntOption(    'OutputFormat', number=MeshFormat.FORMAT_HDF5, name='HDF5')
    CreateIntOption(    'OutputFormat', number=MeshFormat.FORMAT_VTK , name='VTK')
    CreateIntOption(    'OutputFormat', number=MeshFormat.FORMAT_GMSH, name='GMSH')
    CreateLogical(      'DebugVisu'   , default=False , help='Launch the GMSH GUI to visualize the mesh')


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

    io_vars.debugvisu    = GetLogical('DebugVisu')

    # hopout.info('INIT OUTPUT DONE!')


def IO() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Common
    from pyhope.io.io_vars import MeshFormat, ELEM, ELEMTYPE
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('OUTPUT MESH...')

    match io_vars.outputformat:
        case MeshFormat.FORMAT_HDF5:
            mesh  = mesh_vars.mesh
            elems = mesh_vars.elems
            sides = mesh_vars.sides

            nElems = len(elems)
            nSides = len(sides)
            nNodes = sum(s.nodes.size for s in elems)  # number of non-unique nodes

            bcs   = mesh_vars.bcs
            nBCs  = len(bcs)

            pname = io_vars.projectname
            fname = '{}_mesh.h5'.format(pname)

            elemInfo, sideInfo, nodeInfo, nodeCoords, \
            FEMElemInfo, vertexInfo, vertexConnectInfo, edgeInfo, edgeConnectInto, \
            elemCounter = getMeshInfo()

            # Print the final output
            hopout.sep()
            for elemType in ELEM.TYPES:
                if elemCounter[elemType] > 0:
                    hopout.info( ELEMTYPE(elemType) + ': {:12d}'.format(elemCounter[elemType]))
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
                f.attrs['nUniqueSides'  ] = np.max(sideInfo[:, 1])
                f.attrs['nUniqueNodes'  ] = np.max(nodeInfo)

                _ = f.create_dataset('ElemInfo'     , data=elemInfo)
                _ = f.create_dataset('ElemCounter'  , data=np.array(list(elemCounter.items()), dtype=np.int32))
                _ = f.create_dataset('SideInfo'     , data=sideInfo)
                _ = f.create_dataset('GlobalNodeIDs', data=nodeInfo)
                _ = f.create_dataset('NodeCoords'   , data=nodeCoords)

                if FEMElemInfo is not None:
                    f.attrs['FEMconnect'] = 'ON'
                    _ = f.create_dataset('FEMElemInfo'       , data=FEMElemInfo)
                    _ = f.create_dataset('VertexInfo'        , data=vertexInfo)
                    _ = f.create_dataset('VertexConnectInfo' , data=vertexConnectInfo)
                    _ = f.create_dataset('EdgeInfo'          , data=edgeInfo)
                    _ = f.create_dataset('EdgeConnectInfo'   , data=edgeConnectInto)

                # Store boundary information
                f.attrs['nBCs'          ] = nBCs
                bcNames = [f'{s.name:<255}' for s in bcs]
                bcTypes = np.zeros((nBCs, 4), dtype=np.int32)
                for iBC, bc in enumerate(bcs):
                    bcTypes[iBC, :] = bc.type

                _ = f.create_dataset('BCNames'   , data=np.array(bcNames, dtype='S'))
                _ = f.create_dataset('BCType'    , data=bcTypes)

                # Check if there is a periodic vector and write it to mesh file
                nVV = len(mesh_vars.vvs)
                if nVV > 0:
                    vvs = np.zeros((nVV, 3), dtype=np.float64)
                    for iVV, vv in enumerate(mesh_vars.vvs):
                        vvs[iVV, :] = vv['Dir']
                    _ = f.create_dataset('VV', data=vvs)

        case MeshFormat.FORMAT_VTK:
            mesh  = mesh_vars.mesh
            pname = io_vars.projectname
            fname = '{}_mesh.vtk'.format(pname)

            hopout.sep()
            hopout.routine('Writing VTK mesh to "{}"'.format(fname))

            mesh.write(fname, file_format='vtk42')

        case MeshFormat.FORMAT_GMSH:
            mesh  = mesh_vars.mesh
            pname = io_vars.projectname
            fname = '{}_mesh.msh'.format(pname)

            # Mixed elements required gmsh:dim_tags
            # > FIXME: THIS ARE DUMMY ENTRIES AND ONLY GENERATE A POINT MESH
            mesh.point_data.update({'gmsh:dim_tags': np.array([[0, i] for i in range(len(mesh.points))])})

            # Mixed elements require gmsh:physical and gmsh:geometrical
            # > FIXME: THIS ARE DUMMY ENTRIES AND ONLY GENERATE A POINT MESH
            cell_types = mesh.cells_dict.keys()
            cell_data  = [np.ones(mesh.cells_dict[cell_type].data.shape[1], dtype=int) for cell_type in cell_types
]
            mesh.cell_data_dict.update({'gmsh:physical':    cell_data})
            mesh.cell_data_dict.update({'gmsh:geometrical': cell_data})

            hopout.sep()
            hopout.routine('Writing GMSH mesh to "{}"'.format(fname))

            hopout.warning('GMSH output is not yet fully supported, only a point mesh is generated!')

            mesh.write(fname, file_format='gmsh')

        case _:  # Default
            hopout.error('Unknown output format {}, exiting...'.format(io_vars.outputformat))

    # hopout.sep()
    # hopout.info('OUTPUT MESH DONE!')


def getMeshInfo() -> tuple[np.ndarray,         # ElemInfo
                           np.ndarray,         # SideInfo
                           np.ndarray,         # NodeInfo
                           np.ndarray,         # NodeCoords
                           np.ndarray | None,  # Optional[FEMElemInfo]
                           np.ndarray | None,  # Optional[VertexInfo]
                           np.ndarray | None,  # Optional[VertexConnectInfo]
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
    nNodes: Final[int] = np.sum([s.nodes.size for s in elems])  # number of non-unique nodes

    # Create the ElemCounter
    elemCounter = OrderedDict()
    for elemType in ELEM.TYPES:
        elemCounter[elemType] = 0

    # Fill the ElemInfo
    elemInfo  = np.zeros((nElems, ELEM.INFOSIZE), dtype=np.int32)
    sideCount = 0  # elem['Sides'] might work as well
    nodeCount = 0  # elem['Nodes'] contains the unique nodes

    for iElem, elem in enumerate(elems):
        elemInfo[iElem, ELEM.TYPE     ] = elem.type
        elemInfo[iElem, ELEM.ZONE     ] = 1  # FIXME

        elemInfo[iElem, ELEM.FIRSTSIDE] = sideCount
        elemInfo[iElem, ELEM.LASTSIDE ] = sideCount + len(elem.sides)
        sideCount += len(elem.sides)

        elemInfo[iElem, ELEM.FIRSTNODE] = nodeCount
        elemInfo[iElem, ELEM.LASTNODE ] = nodeCount + len(elem.nodes)
        nodeCount += len(elem.nodes)

        elemCounter[elem.type] += 1

    # Set the global side ID
    globalSideID     = 0
    highestSideID    = 0
    usedSideIDs      = set()  # Set to track used side IDs
    availableSideIDs = []     # Min-heap for gap

    for iSide, side in enumerate(sides):
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

        if side.connection is None:         # BC side
            pass
        elif side.connection < 0:           # Big mortar side
            pass
        elif side.MS == 1:                  # Internal / periodic side (master side)
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

    # Fill the SideInfo
    sideInfo  = np.zeros((nSides, SIDE.INFOSIZE), dtype=np.int32)

    for iSide, side in enumerate(sides):
        sideInfo[iSide, SIDE.TYPE     ] = side.sideType
        sideInfo[iSide, SIDE.ID       ] = side.globalSideID
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
            if side.sideType < 0:  # Small mortar side
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

    # Fill the NodeInfo
    nodeInfo   = np.zeros((nNodes)   , dtype=np.int32)

    # Fill the NodeCoords
    nodeCoords = np.zeros((nNodes, 3), dtype=np.float64)
    nodeCount  = 0

    # Cache the mapping
    linCache   = {}

    for iElem, elem in enumerate(elems):
        # Mesh coordinates are stored in meshIO sorting
        elemType = elem.type
        if elemType in linCache:
            mapLin = linCache[elemType]
        else:
            _, mapLin = LINTEN(elem.type, order=mesh_vars.nGeo)
            mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))
            linCache[elemType] = mapLin

        elemNodes  = np.asarray(elem.nodes)
        nElemNodes = elemNodes.size
        indices    = nodeCount + mapLin[:nElemNodes]

        # Assign nodeInfo and nodeCoords in vectorized fashion
        nodeInfo[  indices   ] = elemNodes + 1
        nodeCoords[indices, :] = points[elemNodes]

        nodeCount += nElemNodes

    if hasattr(elems[0], 'vertexInfo') and elems[0].vertexInfo is not None:
        FEMElemInfo, vertexInfo, vertexConnectInfo, edgeInfo, edgeConnectInto = getFEMInfo(nodeInfo)
    else:
        FEMElemInfo, vertexInfo, vertexConnectInfo, edgeInfo, edgeConnectInto = [None for _ in range(5)]

    return elemInfo, sideInfo, nodeInfo, nodeCoords, \
           FEMElemInfo, vertexInfo, vertexConnectInfo, edgeInfo, edgeConnectInto, \
           elemCounter
