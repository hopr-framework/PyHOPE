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
import sys
from dataclasses import dataclass
from functools import cache
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


@dataclass
class ELEM:
    INFOSIZE:  int = 6
    TYPE:      int = 0
    ZONE:      int = 1
    FIRSTSIDE: int = 2
    LASTSIDE:  int = 3
    FIRSTNODE: int = 4
    LASTNODE:  int = 5

    TYPES: tuple[int, ...] = (104, 204, 105, 115, 205, 106, 116, 206, 108, 118, 208)


@dataclass
class SIDE:
    INFOSIZE: int = 5
    TYPE:     int = 0
    ID:       int = 1
    NBELEMID: int = 2
    NBLOCSIDE_FLIP: int = 3
    BCID:     int = 4


@cache
def ELEMTYPE(elemType: int) -> str:
    """ Name of a given element type
    """
    match elemType:
        case 104:
            return ' Straight-edge Tetrahedra '
        case 204:
            return '        Curved Tetrahedra '
        case 105:
            return '  Planar-faced Pyramids   '
        case 115:
            return ' Straight-edge Pyramids   '
        case 205:
            return '        Curved Pyramids   '
        case 106:
            return '  Planar-faced Prisms     '
        case 116:
            return ' Straight-edge Prisms     '
        case 206:
            return '        Curved Prisms     '
        case 108:
            return '  Planar-faced Hexahedra  '
        case 118:
            return ' Straight-edge Hexahedra  '
        case 208:
            return '        Curved Hexahedra  '
        case _:  # Default
            print('Error in ELEMTYPE, unknown elemType')
            sys.exit(1)


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

    hopout.info('INIT OUTPUT DONE!')


def IO() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Common
    from pyhope.io.io_vars import MeshFormat
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
            nNodes = np.sum([s.nodes.size for s in elems])  # number of non-unique nodes

            bcs   = mesh_vars.bcs
            nBCs  = len(bcs)

            pname = io_vars.projectname
            fname = '{}_mesh.h5'.format(pname)

            elemInfo, sideInfo, nodeInfo, nodeCoords, elemCounter = getMeshInfo()

            # Print the final output
            hopout.sep()
            for elemType in ELEM.TYPES:
                if elemCounter[elemType] > 0:
                    hopout.info( ELEMTYPE(elemType) + ': {:12d}'.format(elemCounter[elemType]))
            hopout.sep()
            hopout.routine('Writing HDF5 mesh to "{}"'.format(fname))
            hopout.sep()

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

                _ = f.create_dataset('ElemInfo'     , data=elemInfo)
                _ = f.create_dataset('SideInfo'     , data=sideInfo)
                _ = f.create_dataset('GlobalNodeIDs', data=nodeInfo)
                _ = f.create_dataset('NodeCoords'   , data=nodeCoords)

                # Store boundary information
                f.attrs['nBCs'          ] = nBCs
                bcNames = [f'{s.name:<255}' for s in bcs]
                bcTypes = np.zeros((nBCs, 4), dtype=np.int32)
                for iBC, bc in enumerate(bcs):
                    bcTypes[iBC, :] = bc.type

                _ = f.create_dataset('BCNames'   , data=np.bytes_(bcNames))
                _ = f.create_dataset('BCType'    , data=bcTypes)

        case MeshFormat.FORMAT_VTK:
            mesh  = mesh_vars.mesh
            pname = io_vars.projectname
            fname = '{}_mesh.vtk'.format(pname)

            hopout.routine('Writing VTK mesh to "{}"'.format(fname))

            mesh.write(fname, file_format='vtk42')

        case _:  # Default
            hopout.warning('Unknown output format {}, exiting...'.format(io_vars.outputformat))
            sys.exit(1)

    hopout.info('OUTPUT MESH DONE!')


def getMeshInfo() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_common import LINTEN
    # ------------------------------------------------------

    mesh   = mesh_vars.mesh
    elems  = mesh_vars.elems
    sides  = mesh_vars.sides
    points = mesh.points

    nElems = len(elems)
    nSides = len(sides)
    nNodes = np.sum([s.nodes.size for s in elems])  # number of non-unique nodes

    # Create the ElemCounter
    elemCounter = dict()
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

            # Periodic sides still have a BCID
            if side.bcid is not None:
                sideInfo[iSide, SIDE.BCID      ] = side.bcid + 1
            else:
                sideInfo[iSide, SIDE.BCID      ] = 0

    # Fill the NodeInfo
    nodeInfo   = np.zeros((nNodes)   , dtype=np.int32)

    # Fill the NodeCoords
    nodeCoords = np.zeros((nNodes, 3), dtype=np.float64)
    nodeCount  = 0

    for iElem, elem in enumerate(elems):
        # Mesh coordinates are stored in meshIO sorting
        linMap    = LINTEN(elem.type, order=mesh_vars.nGeo)
        # meshio accesses them in their own ordering
        # > need to reverse the mapping
        mapLin    = {k: v for v, k in enumerate(linMap)}
        elemNodes = elem.nodes

        # Access the actual nodeCoords and reorder them
        for iNode, nodeID in enumerate(elemNodes):
            nodeInfo[  nodeCount + mapLin[iNode]   ] = nodeID + 1
            nodeCoords[nodeCount + mapLin[iNode], :] = points[nodeID]

        nodeCount += len(elemNodes)

    return elemInfo, sideInfo, nodeInfo, nodeCoords, elemCounter
