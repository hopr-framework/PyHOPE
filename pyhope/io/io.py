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
from typing import Tuple
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


class ELEM:
    INFOSIZE  = 6
    TYPE      = 0
    ZONE      = 1
    FIRSTSIDE = 2
    LASTSIDE  = 3
    FIRSTNODE = 4
    LASTNODE  = 5

    TYPES     = [104, 204, 105, 115, 205, 106, 116, 206, 108, 118, 208]


class SIDE:
    INFOSIZE  = 5
    TYPE      = 0
    ID        = 1
    NBELEMID  = 2
    NBLOCSIDE_FLIP = 3
    BCID      = 4


def LINMAP(elemType: int, order: int = 1) -> np.ndarray:
    """ CGNS -> IJK ordering for element corner nodes
    """
    # Local imports ----------------------------------------
    # from pyhope.io.io_cgns import genHEXMAPCGNS
    # from pyhope.io.io_vtk import genHEXMAPVTK
    from pyhope.io.io_meshio import genHEXMAPMESHIO
    # ------------------------------------------------------
    match elemType:
        # Straight-sided elements, hard-coded
        case 104:  # Tetraeder
            return np.array([0, 1, 2, 3])
        case 105:  # Pyramid
            return np.array([0, 1, 3, 2, 4])
        case 106:  # Prism
            return np.array([0, 1, 2, 3, 4, 5])
        case 108:  # Hexaeder
            return np.array([0, 1, 3, 2, 4, 5, 7, 6])
        # Curved elements, use mapping
        case 208:  # Hexaeder
            # # CGNS
            # try:
            #     from pyhope.mesh.mesh_vars import HEXMAP
            # except ImportError:
            #     genHEXMAP(order+1)
            #     from pyhope.mesh.mesh_vars import HEXMAP

            # # VTK
            # try:
            #     from pyhope.mesh.mesh_vars import HEXMAP
            # except ImportError:
            #     genHEXMAPVTK(order+1)
            #     from pyhope.mesh.mesh_vars import HEXMAP

            # MESHIO
            try:
                from pyhope.mesh.mesh_vars import HEXMAP
            except ImportError:
                genHEXMAPMESHIO(order+1)
                from pyhope.mesh.mesh_vars import HEXMAP
            return HEXMAP
        case _:  # Default
            print('Error in LINMAP, unknown elemType')
            sys.exit(1)


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

            # elemInfo, sideInfo, nodeInfo, nodeCoords, elemCounter = getMeshInfo()
            elemInfo, sideInfo, _, nodeCoords, elemCounter = getMeshInfo()

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

                f.create_dataset('ElemInfo'  , data=elemInfo)
                f.create_dataset('SideInfo'  , data=sideInfo)
                # f.create_dataset('NodeInfo'  , data=nodeInfo)
                f.create_dataset('NodeCoords', data=nodeCoords)

                # Store boundary information
                f.attrs['nBCs'          ] = nBCs
                bcNames = [f'{s.name:<255}' for s in bcs]
                bcTypes = np.zeros((nBCs, 4), dtype=np.int32)
                for iBC, bc in enumerate(bcs):
                    bcTypes[iBC, :] = bc.type

                f.create_dataset('BCNames'   , data=np.bytes_(bcNames))
                f.create_dataset('BCType'    , data=bcTypes)

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


def getMeshInfo() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------

    mesh  = mesh_vars.mesh
    elems = mesh_vars.elems
    sides = mesh_vars.sides
    nodes = mesh.points

    nElems = len(elems)
    nSides = len(sides)
    nNodes = np.sum([s.nodes.size for s in elems])  # number of non-unique nodes

    # Create the ElemCounter
    elemCounter = dict()
    for iType, elemType in enumerate(ELEM.TYPES):
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
        if side.connection is not None:
            nbSideID = side.connection
            nbElemID = sides[nbSideID].elemID + 1  # Python -> HOPR index
            sideInfo[iSide, SIDE.NBELEMID      ] = nbElemID
            if side.flip == 0:  # Master side
                sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = sides[nbSideID].locSide*10
            else:
                sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = sides[nbSideID].locSide*10 + side.flip

            # Periodic sides still have a BCID
            if side.bcid is not None:
                sideInfo[iSide, SIDE.BCID      ] = side.bcid + 1
            else:
                sideInfo[iSide, SIDE.BCID      ] = 0
        else:
            sideInfo[iSide, SIDE.NBELEMID      ] = 0
            sideInfo[iSide, SIDE.NBLOCSIDE_FLIP] = 0
            sideInfo[iSide, SIDE.BCID          ] = side.bcid + 1

    # Fill the NodeInfo
    nodeInfo = np.zeros((ELEM.INFOSIZE, nNodes), dtype=np.int32)

    # Fill the NodeCoords
    nodeCoords = np.zeros((nNodes, 3), dtype=np.float64)
    nodeCount  = 0

    for iElem, elem in enumerate(elems):
        # Mesh coordinates are stored in meshIO sorting
        linMap    = LINMAP(elem.type, order=mesh_vars.nGeo)
        # meshio accesses them in their own ordering
        # > need to reverse the mapping
        mapLin    = {k: v for v, k in enumerate(linMap)}
        elemNodes = elem.nodes

        # Access the actual nodeCoords and reorder them
        for iNode, nodeID in enumerate(elemNodes):
            nodeCoords[nodeCount + mapLin[iNode], :] = nodes[nodeID]

        nodeCount += len(elemNodes)

    return elemInfo, sideInfo, nodeInfo, nodeCoords, elemCounter
