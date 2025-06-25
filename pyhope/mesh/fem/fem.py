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
from itertools import chain
from typing import Dict, Tuple, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def copysign_int(x: int, y: int) -> int:
    """ Return a int with the magnitude (absolute value) of x but the sign of y
    """
    # Standard libraries -----------------------------------
    import math
    # ------------------------------------------------------
    return int(math.copysign(x, y))


def FEMConnect() -> None:
    """ Generate connectivity information for edges and vertices
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import CountOption, GetLogical
    from pyhope.mesh.mesh_common import edges, edge_to_corner
    # ------------------------------------------------------

    if CountOption('doFEMConnect') == 0:
        return None

    hopout.separator()
    hopout.info('GENERATE FINITE ELEMENT METHOD (FEM) CONNECTIVITY...')
    hopout.sep()

    doFEMConnect = GetLogical('doFEMConnect')
    if not doFEMConnect:
        hopout.separator()
        return None

    elems     = mesh_vars.elems
    periNodes = mesh_vars.periNodes

    # Create a bidirectional lookup using a single dictionary comprehension with chain
    periDict = { k: v for k, v in chain(((int(node), int(peri)) for (node, _), peri in periNodes.items()),
                                        ((int(peri), int(node)) for (node, _), peri in periNodes.items()))}

    # Build mapping of each node -> set of element indices that include that node.
    nodeToElements = defaultdict(set)
    for idx, elem in enumerate(elems):
        for n in elem.nodes:
            nodeToElements[int(n)].add(idx)

    # Precompute combined connectivity for each node
    # > For a given node, the combined set is:
    # > nodeToElements[node] ∪ nodeToElements[periDict[node]]
    nodeConn = { node: elemSet.union(nodeToElements.get(periDict.get(node), set()))
                 for node, elemSet in nodeToElements.items()}

    # Collect all unique canonical vertices from every element
    # > The canonical vertex is the minimum of the node and its periodic counterpart
    canonicalSet = { min(int(node), periDict.get(int(node), int(node))) for elem in elems
                                                                        for node in elem.nodes[:(elem.type % 100)]}

    # Create a mapping from each canonical vertex to a unique index
    # > FEMVertexID starts at 1
    sortedCanonical = sorted(canonicalSet)
    FEMNodeMapping  = { canonical: newID for newID, canonical in enumerate(sortedCanonical, start=1)}

    # Build the vertex connectivity
    for idx, elem in enumerate(elems):
        vertexInfo: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        for locNode, node in enumerate(int(n) for n in elem.nodes[:elem.type % 100]):
            # Determine canonical vertex id
            canonical   = min(node, periDict.get(node, node))
            FEMVertexID = FEMNodeMapping[canonical]
            # Retrive connectivity set for the node
            nodeVertex = nodeConn.get(node, set())
            vertexInfo[locNode] = (FEMVertexID, tuple(sorted(nodeVertex)))
        # Set the vertex connectivity for the element
        elem.vertexInfo = vertexInfo

        # Build the edge connectivity
        # > Loop over all edges of an element
        # for iEdge, edge in enumerate(edge_to_corner):
        edgeInfo: Dict[int, Tuple[int, int | None, Tuple[int, ...], Tuple[int, ...]]] = {}
        for iEdge in edges(elem.type):
            # Get the nodes of the edge
            edge        = edge_to_corner(iEdge, elem.type)
            edge        = tuple(int(s) for s in elem.nodes[edge])
            # Determine canonical vertex ID
            canonical   = [min(edge[s], cast(int, periDict.get(edge[s], edge[s]))) for s in range(2)]
            # Get the FEM vertex ID
            FEMVertexID = tuple(FEMNodeMapping[c] for c in canonical)
            # Set the edge connectivity for the element
            # > FEMVertexID is unsorted as it contains the global orientation of the edge
            edgeInfo[iEdge] = (iEdge, None, FEMVertexID, edge)
        # Set the edge information for the element
        elem.edgeInfo = edgeInfo


def getFEMInfo(nodeInfo: np.ndarray) -> tuple[np.ndarray,  # FEMElemInfo
                                              np.ndarray,  # VertexInfo
                                              np.ndarray,  # VertexConnectInfo
                                              np.ndarray,  # EdgeInfo
                                              np.ndarray   # EdgeConnectInfo
                                             ]:
    """ Extract the FEM connectivity information and return four arrays

     - FEMElemInfo      : [offsetIndEdge, lastIndEdge, offsetIndVertex, lastIndVertex]
     - vertexInfo       : [FEMVertexID, offsetIndVertexConnect, lastIndVertexConnect]
     - vertexConnectInfo: [nbElemId, nbLocVertexId]
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    elems  = mesh_vars.elems
    nElems = len(elems)

    # Check if elements contain FEM connectivity
    if not hasattr(elems[0], 'vertexInfo') or elems[0].vertexInfo is None:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Vertex connectivity info ---------------------------------------------------
    # > Build list of all vertex occurrences, appearing in the same order as the elements
    occList = [(FEMVertexID, elemID, locNode) for elemID , elem             in enumerate(elems)  # noqa: E272
                                              for locNode, (FEMVertexID, _) in elem.vertexInfo.items()]

    # > Build mapping from FEM vertex ID to list of occurrences
    groups = defaultdict(list)
    for occIdx, (FEMVertexID, elemID, locNode) in enumerate(occList):
        groups[FEMVertexID].append((occIdx, elemID, locNode))

    # Initialize FEM element information
    FEMElemInfo    = np.zeros((nElems, 4), dtype=np.int32)

    vertexInfoList = []  # List: [FEMVertexID, offsetIndVertexConnect, lastIndVertexConnect]
    vertexConnList = []  # List: [[nbElemID, nbLocVertexID]]
    vertexOffset   = 0
    occGlobalIdx   = 0   # global index in occList

    for elemID, elem in enumerate(elems):
        # Process vertex occurrences for the current element
        for _ in elem.vertexInfo:
            # Get the occurrence information from the global occList
            FEMVertexID, _, locNode = occList[occGlobalIdx]
            groupOcc = groups[FEMVertexID]
            offset   = len(vertexConnList)

            # Identify the master occurrence (lowest occIdx from the occurrence group)
            masterOcc = min(x[0] for x in groupOcc)

            # Build connectivity list for current element, excluding itself
            connections = [(nbElem+1, nbLocal+1) if   otherOcc == masterOcc else (-(nbElem+1), nbLocal+1)  # noqa: E271
                                                 for (otherOcc, nbElem, nbLocal) in groupOcc if otherOcc != occGlobalIdx]

            if connections:
                lastIndex = offset + len(connections)
                vertexConnList.extend(connections)
            else:  # No connections
                lastIndex = offset

            # Append vertex information
            vertexInfoList.append([FEMVertexID, offset, lastIndex])
            occGlobalIdx += 1

        # Set the vertex connectivity offset for this element.
        FEMElemInfo[elemID, 2] = vertexOffset
        FEMElemInfo[elemID, 3] = vertexOffset + len(elem.vertexInfo)
        vertexOffset += len(elem.vertexInfo)

    # Edge   connectivity info ---------------------------------------------------
    edgeInfoList   = []  # List: [FEMEdgeID  , offsetIndEdgeConnect  , lastIndEdgeConnect]
    edgeConnList   = []  # List: [[nbElemID, nbLocEdgeID]]
    edgGlobalIdx   = 0   # global edge index

    for elemID, elem in enumerate(elems):
        # Process edge occurrences for the current element
        for iEdge, (locEdge, edgeIdx, edge, edgeNodes) in enumerate(elem.edgeInfo.values()):
            # Get the elements connected to both edge nodes
            groupOcc  = [groups[e] for e in edge]
            offset    = len(edgeConnList)

            # Identify elements connected to the current edge
            edgeElems = set([e[1]  for g in groupOcc for e in g])  # noqa: E272

            # Build connectivity list for current element, excluding the current edge
            connections = [(nbElem, nbEdgeIdx, nbLocEdge, nbEdge, nbEdgeNodes)
                           for nbElem                                      in edgeElems                            # noqa: E272
                           for (nbLocEdge, nbEdgeIdx, nbEdge, nbEdgeNodes) in elems[nbElem].edgeInfo.values()      # noqa: E272
                           if set(nbEdge) == set(edge) and (nbLocEdge != locEdge or nbElem != elem.elemID)]

            if connections:
                # Sanity check, all edge should have the same global index
                if ((edgeIdx is     None and any(     e  is not None for (_, e, _, _, _) in connections)) or       # noqa: E271, E272
                    (edgeIdx is not None and len({abs(e)             for (_, e, _, _, _) in connections}) != 1)):  # noqa: E271, E272
                    hopout.error('FEMConnect: Inconsistent edge global index', traceback=True)

            # Check if the edges already have an global index
            if edgeIdx is not None:
                # Check if the current edge is the master edge
                if edgeIdx > 0:  # master edge
                    masterID, masterEdge, masterEdgeNodes = -1, edge, edgeNodes
                else:            # slave edge
                    # Find the master edge among the connections
                    masterID = [i for i in range(len(connections)) if connections[i][1] > 0]
                    if len(masterID) != 1:
                        hopout.error('FEMConnect: Inconsistent edge global index', traceback=True)
                    masterID = masterID[0]
                    masterID, masterEdge, masterEdgeNodes = masterID, *connections[masterID][3:5]
            # Otherwise, the current edge is the master edge and the others are slave edges
            else:
                edgGlobalIdx   +=  1
                edgeIdx         = edgGlobalIdx
                masterID, masterEdge, masterEdgeNodes = -1, edge, edgeNodes
                # Set the current edge as master edge
                elem.edgeInfo[iEdge] = (locEdge, edgGlobalIdx, edge, edgeNodes)
                # Set the global index for the other edges
                for nbElem, _, nbLocEdge, _, _ in connections:
                    e = list(elems[nbElem].edgeInfo[nbLocEdge])
                    e[1] = -edgGlobalIdx
                    elems[nbElem].edgeInfo[nbLocEdge] = tuple(e)

            # TODO: Check if the orientation of the master edge is with ascending nodeInfo index
            orientation =  1 if nodeInfo[masterEdgeNodes[0]] < nodeInfo[masterEdgeNodes[1]] else -1
            # Current edge is a slave edge, check our relative orientation
            if masterID != -1:
                # Check if the edge is oriented in the same direction
                orientation = orientation if masterEdge[0] == edge[0] else -orientation

            edgeConn = []
            for iConn, (nbElem, _, nbLocEdge, nbEdge, _) in enumerate(connections):
                # The current edge is the master
                if masterID == -1:
                    orientedElemID  = -(nbElem   +1)
                    orientedLocEdge =   nbLocEdge+1 if nbEdge   == masterEdge               else -(nbLocEdge+1)  # noqa: E272

                # The master edge is one of the connections, indicated by masterID
                else:
                    orientedElemID  =   nbElem   +1 if masterID == iConn                    else -(nbElem   +1)  # noqa: E272
                    orientedLocEdge =   nbLocEdge+1 if nbEdge   == connections[masterID][3] else -(nbLocEdge+1)

                # Append the edge connectivity
                edgeConn.append([orientedElemID, orientedLocEdge])

            if connections:
                lastIndex = offset + len(edgeConn)
                edgeConnList.extend(edgeConn)
            else:  # No connections
                lastIndex = offset

            # Append edge information
            edgeInfoList.append([copysign_int(edgeIdx, orientation), offset, lastIndex])

    # INFO: Same output as above but looping over the occurrences
    # for occIdx, (FEMVertexId, elemID, locNode) in enumerate(occList):
    #     groupOcc    = groups[FEMVertexId]
    #     offset      = len(vertexConnList)
    #
    #     # Identify the master occurrence (lowest occurrence index in the group)
    #     masterOcc   = min(x[0] for x in groupOcc)
    #
    #     # Build connectivity list for current element, excluding itself
    #     connections = [(nbElem+1, nbLocal+1) if otherOcc == masterOcc else (-(nbElem+1), nbLocal+1)
    #                    for (otherOcc, nbElem, nbLocal) in groupOcc if otherOcc != occIdx]
    #     if connections:
    #         lastIndex = offset + len(connections)
    #         vertexConnList.extend(connections)
    #     else:  # No connections
    #         lastIndex = offset
    #     # Append vertex information
    #     vertexInfoList.append([FEMVertexId, offset, lastIndex])
    #
    #     # Update FEMElemInfo for the corresponding element.
    #     # Set the offset to the minimum and last index to the maximum among occurrences.
    #     if offset < FEMElemInfo[elemID, 2]:
    #         FEMElemInfo[elemID, 2] = offset
    #     if lastIndex > FEMElemInfo[elemID, 3]:
    #         FEMElemInfo[elemID, 3] = lastIndex

    # Convert lists to numpy arrays
    vertexInfo = np.array(vertexInfoList, dtype=np.int32)
    vertexConn = np.array(vertexConnList, dtype=np.int32) if vertexConnList else np.array((0, 2), dtype=np.int32)

    edgeInfo   = np.array(edgeInfoList  , dtype=np.int32)
    edgeConn   = np.array(edgeConnList  , dtype=np.int32) if edgeConnList   else np.array((0, 2), dtype=np.int32)  # noqa: E272

    return FEMElemInfo, vertexInfo, vertexConn, edgeInfo, edgeConn
