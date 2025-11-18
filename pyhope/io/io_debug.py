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
from functools import cache
from typing import Final, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# Monkey-patching meshio.xdmf.main.XdmfWriter
from pyhope.io.io_xdmf import XdmfWriterInit
meshio.xdmf.main.XdmfWriter.__init__ = XdmfWriterInit  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
# ==================================================================================================================================


# def writeVTM(filename: str,
#              blocks  : Union[list, tuple]) -> None:
#     # Standard libraries -----------------------------------
#     import xml.etree.ElementTree as ET
#     # ------------------------------------------------------
#     # blocks is a list of tuples: (index, name, filepath)
#     vtkfile = ET.Element('VTKFile', attrib={'type'      : 'vtkMultiBlockDataSet',
#                                             'version'   : '1.1',
#                                             'byte_order': 'LittleEndian'})
#
#     multiblock = ET.SubElement(vtkfile, 'vtkMultiBlockDataSet')
#
#     for idx, name, filepath in blocks:
#         ET.SubElement(multiblock, 'DataSet', attrib={'index': str(idx),
#                                                      'name' : name,
#                                                      'file' : filepath})
#
#     tree = ET.ElementTree(vtkfile)
#     tree.write(filename, encoding='utf-8', xml_declaration=True)
@cache
def isValidInt(s) -> bool:
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


def DebugIO() -> None:
    """ Routine to output the debug mesh. Downcast the existing
        PyHOPE format to first order, enrich with debug information
        and output
    """
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_common import edges as ELEMEDGES
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    # hopout.sep()

    mesh   : Final             = mesh_vars.mesh
    mpoints: Final[np.ndarray] = mesh.points
    melems : Final[list]       = mesh_vars.elems
    msides : Final[list]       = mesh_vars.sides
    bcs    : Final[list]       = mesh_vars.bcs
    pname  : Final[str]        = io_vars.projectname

    # Create empty meshio objects
    points    = set()
    elems     = {}
    elemtypes = set()
    sides     = {}
    sidetypes = set()
    nodes     = {}
    edges     = {}

    # Instantiate ELEMTYPE
    elemTypeClass = ELEMTYPE()

    # Ordered index maps for element and side types
    tInv: dict[str, int] = {}
    sInv: dict[str, int] = {}

    # Loop over all elements
    for melem in melems:
        # Correct ElemType for NGeo = 1
        elemNum  = melem.type % 100
        elemType = elemTypeClass.inam[elemNum + 100]
        elemType = ''.join(elemType) if isinstance(elemType, list) else elemType
        if elemType not in tInv:
            tInv[elemType] = len(elemtypes)
            elemtypes.add(elemType)

        # Add the first-order nodes to the points set
        points.update(set(cast(np.ndarray, melem.nodes)[:elemNum]))

        # Add the first-order sides to the sides set
        for sideID in melem.sides:  # ty: ignore [not-iterable]
            # Only consider boundary sides
            if msides[sideID].bcid is not None:
                sideType = 'triangle' if msides[sideID].sideType == 3 else 'quad'
                if sideType not in sInv:
                    sInv[sideType] = len(sidetypes)
                    sidetypes.add(sideType)

    # Create ordered mapping from first-order points to high-order points
    points = list(points)
    pMap   = np.unique(np.array(points))
    pInv   = dict(zip(pMap, range(len(pMap))))

    hasIJK = True if hasattr(mesh_vars, 'nElemsIJK' ) and mesh_vars.nElemsIJK  is not None else False  # noqa: E272
    hasFEM = True if hasattr(melems[0], 'vertexInfo') and melems[0].vertexInfo is not None else False

    # Prepare element and side containers
    for t in elemtypes:
        elems.setdefault(t, [])
    for st in sidetypes:
        sides.setdefault(st, [])
    if hasFEM:
        nodes.setdefault('vertex', [])
        edges.setdefault('line'  , [])

    # Create ordered mapping from first-order elems to high-order elems
    types  = list(elemtypes)
    # Create ordered mapping from first-order sides to high-order sides
    sypes  = list(sidetypes)

    elemdata: dict[str, list] = {'ElemID'  : [list() for _ in range(len(types))],
                                 'ElemType': [list() for _ in range(len(types))],
                                 'ElemZone': [list() for _ in range(len(types))],
                                }
    # (Optional:) Add Jacobians
    if melems and (getattr(melems[0], 'jacobian', None) is not None):
        elemdata.update({'ElemJacobian': [list() for _ in range(len(types))]})
    # (Optional:) Add IJK sorting
    if hasIJK:
        elemdata.update({'Elem_I'      : [list() for _ in range(len(types))]})
        elemdata.update({'Elem_J'      : [list() for _ in range(len(types))]})
        elemdata.update({'Elem_K'      : [list() for _ in range(len(types))]})

    sidedata: dict[str, list] = {'ElemID'  : [list() for _ in range(len(sypes))],
                                 'BCID'    : [list() for _ in range(len(sypes))],
                                 'BCType'  : [list() for _ in range(len(sypes))],
                                 'BCState' : [list() for _ in range(len(sypes))],
                                 'BCAlpha' : [list() for _ in range(len(sypes))],
                                }

    nodedata: dict[str, list] = {}
    edgedata: dict[str, list] = {}
    if hasFEM:
        edgedata.update(                       {'FEMEdgeID'  : [],
                                                'LocEdge'    : [],
                                               })
        # Fully create the nodes here
        nodes = cast(dict[str, npt.ArrayLike], {'vertex':      [np.asarray([s])  for s in range(len(pMap))]})  # noqa: E272
        nodedata.update(                       {'FEMVertexID': [-1 for _ in range(len(pMap))],
                                               })

    # Populate connectivity and data
    for melem in melems:
        # Correct ElemType for NGeo = 1
        elemNum  = melem.type % 100
        elemType = elemTypeClass.inam[elemNum + 100]
        elemType = ''.join(elemType) if isinstance(elemType, list) else elemType
        elemZone = int(melem.zone) if (melem.zone is not None and isValidInt(melem.zone)) else 1
        tidx     = tInv[elemType]

        elemNodes = np.fromiter((pInv[s] for s in cast(np.ndarray, melem.nodes)[:elemNum]), dtype=np.int64, count=elemNum)
        elems[elemType].append(elemNodes)

        # Add the elemData
        elemdata['ElemID'  ][tidx].append(melem.elemID + 1)
        elemdata['ElemType'][tidx].append(melem.type)
        elemdata['ElemZone'][tidx].append(elemZone)
        if 'ElemJacobian' in elemdata:
            elemdata['ElemJacobian'][tidx].append(melem.jacobian)
        if hasIJK:
            elemdata['Elem_I'      ][tidx].append(cast(np.ndarray, melem.elemIJK)[0])
            elemdata['Elem_J'      ][tidx].append(cast(np.ndarray, melem.elemIJK)[1])
            elemdata['Elem_K'      ][tidx].append(cast(np.ndarray, melem.elemIJK)[2])

        # Add the side[Data]
        for sideID in melem.sides:  # ty: ignore [not-iterable]
            # Only consider boundary sides
            side = msides[sideID]
            if side.bcid is not None:
                sideType = 'triangle' if side.sideType == 3 else 'quad'
                sidx     = sInv[sideType]

                # Add the side
                sideNodes = np.fromiter((pInv[s] for s in cast(np.ndarray, side.corners)), dtype=np.int64)
                sides[sideType].append(sideNodes)

                # Add the sideData
                bcID = side.bcid
                bc   = bcs[bcID]
                sidedata['ElemID'  ][sidx].append(melem.elemID + 1)
                sidedata['BCID'    ][sidx].append(bcID         + 1)
                sidedata['BCType'  ][sidx].append(bc.type[0]      )  # ty: ignore [non-subscriptable]
                sidedata['BCState' ][sidx].append(bc.type[2]      )  # ty: ignore [non-subscriptable]
                sidedata['BCAlpha' ][sidx].append(bc.type[3]      )  # ty: ignore [non-subscriptable]

        if hasFEM:
            # Create the FEM vertices
            for locNode, node in enumerate(elemNodes):
                # Add the nodeData
                nodedata['FEMVertexID'][node] = cast(dict, melem.vertexInfo)[locNode][0]  # pyright: ignore[reportPossiblyUnboundVariable]

            # Create the FEM edges
            elemEdges = ELEMEDGES(elemType)
            for edge in elemEdges:
                # Add the edge
                edgeInfo  = cast(dict, melem.edgeInfo)[edge]
                edgeNodes = np.fromiter((pInv[s] for s in edgeInfo[3]), dtype=np.int64)
                edges['line'].append(edgeNodes)

                edgedata['FEMEdgeID'  ].append(edgeInfo[1])
                edgedata['LocEdge'    ].append(edgeInfo[0])

    # Update points to unique first-order coords
    coords = mpoints[pMap]
    # Find the mapping from the cell keys to the elemtypes
    elemOrder = [tInv[cb] for cb in elems.keys()]

    # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
    elemdata = {k: [np.asarray(v[idx]) for idx in elemOrder] for k, v in elemdata.items()}
    eleminfo = {'name': 'Volume'}
    # Clean-up for memory safety
    del elemOrder

    # Create the output list
    debugOut   = []

    # Create the final debugElem with first-order elements
    debugElem  = meshio.Mesh(points    = coords,     # noqa: E251
                             cells     = elems,      # noqa: E251
                             cell_data = elemdata,   # noqa: E251
                             info      = eleminfo,   # noqa: E251
                            )
    debugOut.append(debugElem)
    # fname = f'{pname}_Debug.xdmf'
    # hopout.routine(f'Writing volume  debug mesh to "{fname}"')
    # debugElem.write(fname)
    # # Clean-up for memory safety
    # del debugElem

    # Find the mapping from the side keys to the elemtypes
    sideOrder = [sInv[cb] for cb in sides.keys()]

    # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
    sidedata = {k: [np.asarray(v[idx]) for idx in sideOrder] for k, v in sidedata.items()}
    sideinfo = {'name': 'Surface'}
    # Clean-up for memory safety
    del sideOrder

    # Create the final debugSide with first-order elements
    debugSide  = meshio.Mesh(points    = coords,     # noqa: E251
                             cells     = sides,      # noqa: E251
                             cell_data = sidedata,   # noqa: E251
                             info      = sideinfo,   # noqa: E251
                            )
    debugOut.append(debugSide)
    # fname = f'{pname}_Debug.xdmf'
    # hopout.routine(f'Writing surface debug mesh to "{fname}"')
    # debugSide.write(fname)
    # # Clean-up for memory safety
    # del debugSide
    # del fname

    if hasFEM:
        # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
        edgedata = {k: [np.asarray(v)] for k, v in edgedata.items()}
        edgeinfo = {'name': 'FEMEdges'}

        debugEdge  = meshio.Mesh(points    = coords,     # noqa: E251
                                 cells     = edges,      # noqa: E251
                                 cell_data = edgedata,   # noqa: E251
                                 info      = edgeinfo,   # noqa: E251
                                )
        debugOut.append(debugEdge)
        # fname = f'{pname}_DebugEdge.vtu'
        # hopout.routine(f'Writing edge    debug mesh to "{fname}"')
        # debugNode.write(fname)
        # # Clean-up for memory safety
        # del debugEdge
        # del fname

        # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
        nodedata = {k: [np.asarray(v)] for k, v in nodedata.items()}
        nodeinfo = {'name': 'FEMVertices'}

        debugNode  = meshio.Mesh(points    = coords,     # noqa: E251
                                 cells     = nodes,      # noqa: E251
                                 cell_data = nodedata,   # noqa: E251
                                 info      = nodeinfo,   # noqa: E251
                                )
        debugOut.append(debugNode)
        # fname = f'{pname}_DebugNode.vtu'
        # hopout.routine(f'Writing vertex  debug mesh to "{fname}"')
        # debugNode.write(fname)
        # # Clean-up for memory safety
        # del debugNode
        # del fname

    fname = f'{pname}_DebugMesh.xdmf'
    hopout.routine(f'Writing XDMF mesh to "{fname}"')
    meshio.xdmf.main.XdmfWriter(fname, debugOut)  # ty: ignore[unresolved-attribute]

    # (Optional:) Write wrapper for multiblock file
    # blocks = [(0, 'VolumeMesh' , f'{pname}_DebugElem.vtu'),
    #           (1, 'SurfaceMesh', f'{pname}_DebugSide.vtu')]
    #
    # writeVTM(f'{pname}_DebugMesh.vtm',
    #          blocks)
