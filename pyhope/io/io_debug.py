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
from functools import cache
from typing import Any, Final, Optional
from typing import cast
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
meshio.xdmf.main.XdmfWriter.__init__ = XdmfWriterInit  # pyright: ignore[reportAttributeAccessIssue]
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
def isValidInt(s: Any) -> bool:  # noqa: ANN401
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


def FillElemData(melems: list,
                 elems : dict,
                 hasIJK: bool,
                 # hasFEM: bool,
                 types : list,
                 tInv  : dict[str, int],
                 pMap  : npt.NDArray,
                ) -> tuple[dict, dict[str, list]]:
    # Local imports ----------------------------------------
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    # Instantiate ELEMTYPE
    elemTypeClass = ELEMTYPE()

    # Create a defaultdict since we have optional members
    elemdata = defaultdict(lambda: [[] for _ in range(len(types))])

    # Populate connectivity and data
    for melem in melems:
        # Correct ElemType for NGeo = 1
        elemNum  = melem.type % 10
        elemType = elemTypeClass.inam[elemNum + 100]
        elemType = ''.join(elemType) if isinstance(elemType, list) else elemType
        elemZone = int(melem.zone) if (melem.zone is not None and isValidInt(melem.zone)) else 1
        tidx     = tInv[elemType]

        # pMap is already sorted
        elemNodes = np.searchsorted(pMap, cast(np.ndarray, melem.nodes)[:elemNum])
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

    return elems, dict(elemdata)


def FillSideData(msides: list,
                 sides : dict,
                 sypes : list,
                 sInv  : dict[str, int],
                 pMap  : npt.NDArray,
                 bcs   : list,
                 errOut: bool = False,
                ) -> tuple[dict, dict[str, list]]:

    # Create a defaultdict since we have optional members
    sidedata = defaultdict(lambda: [[] for _ in range(len(sypes))])

    # Populate connectivity and data
    for side in tuple(s for s in msides if s.bcid is not None or errOut):
        sideType = 'triangle' if side.sideType == 3 else 'quad'
        sidx     = sInv[sideType]

        # Add the side
        sideNodes = np.searchsorted(pMap, np.asarray(side.corners))
        sides[sideType].append(sideNodes)

        # Add the sideData
        sidedata['ElemID'  ][sidx].append(side.elemID + 1)

        # Add the boundary dataa
        if side.bcid is not None:
            bcID = side.bcid
            bc   = bcs[bcID]
            sidedata['BCID'    ][sidx].append(bcID       + 1)
            sidedata['BCType'  ][sidx].append(bc.type[0]    )
            sidedata['BCState' ][sidx].append(bc.type[2]    )
            sidedata['BCAlpha' ][sidx].append(bc.type[3]    )

    return sides, dict(sidedata)


def FillEdgeData(melems: list,
                 edges : dict,
                 hasFEM: bool,
                 pMap  : npt.NDArray,
                ) -> tuple[Optional[dict], Optional[dict[str, list]]]:
    # Local imports ----------------------------------------
    from pyhope.mesh.mesh_common import edges as ELEMEDGES
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    if not hasFEM:
        return edges, None

    # Instantiate ELEMTYPE
    elemTypeClass = ELEMTYPE()

    # Create a defaultdict since we have optional members
    edgedata = defaultdict(list)

    # Populate connectivity and data
    for melem in melems:
        # Correct ElemType for NGeo = 1
        elemNum  = melem.type % 10
        elemType = elemTypeClass.inam[elemNum + 100]
        elemType = ''.join(elemType) if isinstance(elemType, list) else elemType

        # Create the FEM edges
        elemEdges = ELEMEDGES(elemType)
        for edge in elemEdges:
            # Add the edge
            edgeInfo  = cast(dict, melem.edgeInfo)[edge]
            edgeNodes = np.searchsorted(pMap, np.asarray(edgeInfo[3]))
            edges['line'].append(edgeNodes)

            edgedata['FEMEdgeID'  ].append(edgeInfo[1])
            edgedata['LocEdge'    ].append(edgeInfo[0])

    return edges, dict(edgedata)


def FillNodeData(melems: list,
                 nodes : dict,
                 hasFEM: bool,
                 pMap  : npt.NDArray,
                ) -> tuple[Optional[dict], Optional[dict[str, list]]]:

    if not hasFEM:
        return nodes, None

    # Fully create the nodes here
    nodes = cast(dict[str, npt.ArrayLike], {'vertex': [np.asarray([s])  for s in range(len(pMap))]})  # noqa: E272
    # Create a defaultdict since we have optional members
    nodedata = defaultdict(lambda: [1 for _ in range(len(pMap))])

    # Populate connectivity and data
    for melem in melems:
        # Correct ElemType for NGeo = 1
        elemNum  = melem.type % 10
        # pMap is already sorted
        elemNodes = np.searchsorted(pMap, cast(np.ndarray, melem.nodes)[:elemNum])

        # Create the FEM vertices
        for locNode, node in enumerate(elemNodes):
            # Add the nodeData
            nodedata['FEMVertexID'][node] = cast(dict, melem.vertexInfo)[locNode][0]

    return nodes, dict(nodedata)


def DebugIO(errElems: Optional[list] = None,
            errSides: Optional[list] = None) -> None:
    """ Routine to output the debug mesh. Downcast the existing
        PyHOPE format to first order, enrich with debug information
        and output in XDMF format
    """
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    # Write a low-order debug mesh if requested
    if not io_vars.debugmesh:
        return None

    mesh   : Final             = mesh_vars.mesh
    mpoints: Final[np.ndarray] = mesh.points
    melems : Final[list]       = mesh_vars.elems
    msides : Final[list]       = mesh_vars.sides
    bcs    : Final[list]       = mesh_vars.bcs
    pname  : Final[str]        = io_vars.projectname

    # Convert error lists to sets
    errElems = set(errElems) if errElems is not None else ()
    errSides = set(errSides) if errSides is not None else ()

    # Create empty meshio objects
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
        elemNum  = melem.type % 10
        elemType = elemTypeClass.inam[elemNum + 100]
        elemType = ''.join(elemType) if isinstance(elemType, list) else elemType
        if elemType not in tInv:
            tInv[elemType] = len(elemtypes)
            elemtypes.add(elemType)

        # Add the first-order sides to the sides set
        for sideID in (s for s in melem.sides if msides[s].bcid is not None or s in (errSides if errSides is not None else ())):  # ty: ignore [not-iterable]
            # Only consider boundary/error sides
            sideType = 'triangle' if msides[sideID].sideType == 3 else 'quad'
            if sideType not in sInv:
                sInv[sideType] = len(sidetypes)
                sidetypes.add(sideType)

    # Create ordered mapping from first-order points to high-order points
    points = np.concatenate([np.asarray(melem.nodes)[:melem.type % 10] for melem in melems])
    pMap   = np.unique(points)

    hasIJK = bool(hasattr(mesh_vars, 'nElemsIJK')  and mesh_vars.nElemsIJK  is not None)  # noqa: E272
    hasFEM = bool(hasattr(melems[0], 'vertexInfo') and melems[0].vertexInfo is not None)

    # Prepare element and side containers
    for t in elemtypes:
        elems.setdefault(t, [])
    for st in sidetypes:
        sides.setdefault(st, [])
    if hasFEM:
        edges.setdefault('line'  , [])

    # Create ordered mapping from first-order elems to high-order elems
    types  = list(elemtypes)
    # Create ordered mapping from first-order sides to high-order sides
    sypes  = list(sidetypes)

    # Fill elem data
    elems, elemdata = FillElemData(melems, elems, hasIJK, types, tInv, pMap)
    # Fill the side data (optional)
    sides, sidedata = FillSideData(msides, sides,         sypes, sInv, pMap, bcs, bool(errSides) and not any(s.bcid is not None for s in msides))  # ruff: ignore[line-too-long]
    # Fill the edge data (optional)
    edges, edgedata = FillEdgeData(melems, edges, hasFEM,              pMap)
    # Fill the node data (optional)
    nodes, nodedata = FillNodeData(melems, nodes, hasFEM,              pMap)

    # Create the output list
    debugOut   = []

    # Update points to unique first-order coords
    coords = mpoints[pMap]
    # Find the mapping from the cell keys to the elemtypes
    elemOrder = [tInv[cb] for cb in elems]

    # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
    elemdata = {k: [np.asarray(v[idx]) for idx in elemOrder] for k, v in elemdata.items()}
    eleminfo = {'name': 'Volume'}

    # Create the final debugElem with first-order elements
    debugElem  = meshio.Mesh(points    = coords,     # noqa: E251
                             cells     = elems,      # noqa: E251
                             cell_data = elemdata,   # noqa: E251
                             info      = eleminfo,   # noqa: E251
                            )
    debugOut.append(debugElem)

    if len(errElems):  # pragma: no cover
        # Create empty meshio objects
        errorElems = {}

        # Prepare element and side containers
        for t in elemtypes:
            errorElems.setdefault(t, [])

        # Fill error elem data
        errorElems, errorElemdata = FillElemData((e for e in melems if e.elemID in errElems), errorElems, hasIJK, types, tInv, pMap)  # ruff: ignore[line-too-long]

        # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
        errorElemdata = {k: [np.asarray(v[idx]) for idx in elemOrder] for k, v in errorElemdata.items()}
        errorEleminfo = {'name': 'Volume [Error]'}

        # Create the final debugElem with first-order elements
        debugElem  = meshio.Mesh(points    = coords,          # noqa: E251
                                 cells     = errorElems,      # noqa: E251
                                 cell_data = errorElemdata,   # noqa: E251
                                 info      = errorEleminfo,   # noqa: E251
                                )
        debugOut.append(debugElem)

    # Clean-up for memory safety
    del elemOrder

    # Find the mapping from the side keys to the elemtypes
    sideOrder = [sInv[cb] for cb in sides]

    # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
    sidedata = {k: [np.asarray(v[idx]) for idx in sideOrder] for k, v in sidedata.items()}
    sideinfo = {'name': 'Surface'}

    # Create the final debugSide with first-order elements
    debugSide  = meshio.Mesh(points    = coords,     # noqa: E251
                             cells     = sides,      # noqa: E251
                             cell_data = sidedata,   # noqa: E251
                             info      = sideinfo,   # noqa: E251
                            )
    debugOut.append(debugSide)

    if len(errSides):  # pragma: no cover
        # Create empty meshio objects
        errorSides = {}

        # Prepare element and side containers
        for st in sidetypes:
            errorSides.setdefault(st, [])

        # Fill the error side data
        errorSides, errorSidedata = FillSideData((s for s in msides if s.sideID in errSides), errorSides,         sypes, sInv, pMap, bcs, True)  # ruff: ignore[line-too-long]

        # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
        errorSidedata = {k: [np.asarray(v[idx]) for idx in sideOrder] for k, v in errorSidedata.items()}
        errorSideinfo = {'name': 'Surface [Error]'}

        # Create the final debugSide with first-order elements
        debugSide  = meshio.Mesh(points    = coords,          # noqa: E251
                                 cells     = errorSides,      # noqa: E251
                                 cell_data = errorSidedata,   # noqa: E251
                                 info      = errorSideinfo,   # noqa: E251
                                )
        debugOut.append(debugSide)

    # Clean-up for memory safety
    del sideOrder

    if edgedata is not None:
        # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
        edgedata = {k: [np.asarray(v)] for k, v in edgedata.items()}
        edgeinfo = {'name': 'FEMEdges'}

        debugEdge  = meshio.Mesh(points    = coords,     # noqa: E251
                                 cells     = edges,      # noqa: E251
                                 cell_data = edgedata,   # noqa: E251
                                 info      = edgeinfo,   # noqa: E251
                                )
        debugOut.append(debugEdge)

    if nodedata is not None:
        # Ensure cell_data lists are aligned to the actual cell block order used by meshio.Mesh
        nodedata = {k: [np.asarray(v)] for k, v in nodedata.items()}
        nodeinfo = {'name': 'FEMVertices'}

        debugNode  = meshio.Mesh(points    = coords,     # noqa: E251
                                 cells     = nodes,      # noqa: E251
                                 cell_data = nodedata,   # noqa: E251
                                 info      = nodeinfo,   # noqa: E251
                                )
        debugOut.append(debugNode)

    fname = f'{pname}_DebugMesh.xdmf'
    hopout.routine(f'Writing XDMF mesh to "{fname}"')
    meshio.xdmf.main.XdmfWriter(fname, debugOut)

    if len(errElems):  # pragma: no cover
        print('│' + hopout.Symbols.INFO[:3] +
              f'Reason: Detected {len(errElems)} / {len(melems)} erroneous elements, written to "Volume [Error]"')
    if len(errSides):  # pragma: no cover
        print('│' + hopout.Symbols.INFO[:3] +
              f'Reason: Detected {len(errSides)} / {len(msides)} erroneous sides, written to "Surface [Error]"')
