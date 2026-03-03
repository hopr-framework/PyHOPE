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
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Tuple, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


# def copysign_int(x: int, y: int) -> int:
#     """ Return a int with the magnitude (absolute value) of x but the sign of y
#     """
#     # Standard libraries -----------------------------------
#     import math
#     # ------------------------------------------------------
#     return int(math.copysign(x, y))


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

    if not GetLogical('doFEMConnect'):
        hopout.separator()
        return None

    elems     = mesh_vars.elems
    periNodes = mesh_vars.periNodes

    # Build a graph of all periodic connections across all BCs
    periGraph: dict[int, set[int]] = defaultdict(set)
    for (node, _), peri in periNodes.items():
        node, peri = int(node), int(peri)
        periGraph[node].add(peri)
        periGraph[peri].add(node)

    # Find connected components; representative = min(node indices in component)
    nodeFirst: dict[int, int]      = {}  # node -> representative (canonical)
    nodeGroup: dict[int, set[int]] = {}  # representative -> full member set

    for nodeStart in list(periGraph.keys()):
        if nodeStart in nodeFirst:
            continue

        stack     = [nodeStart]
        component = set()

        while stack:
            currentNode = stack.pop()
            if currentNode in component:
                continue

            component.add(currentNode)
            for nxt in periGraph[currentNode]:
                if nxt not in component:
                    stack.append(nxt)

        canonical_rep = min(component)
        for v in component:
            nodeFirst[v] = canonical_rep

        nodeGroup[canonical_rep] = component

    # Build mapping of each node -> set of element indices that include that node
    nodeToElements = defaultdict(set)
    for elemID, elem in enumerate(elems):
        for node in cast(np.ndarray, elem.nodes)[:cast(int, elem.type) % 10]:
            nodeToElements[int(node)].add(elemID)

    # Precompute combined connectivity for each node
    nodeConn = {}
    for members in nodeGroup.values():
        combined = set().union(*(nodeToElements[m] for m in members if m in nodeToElements))
        for nodeMap in members:
            nodeConn[nodeMap] = combined

    for node, elemSet in nodeToElements.items():
        if node not in nodeConn:
            nodeConn[node] = elemSet

    # Collect all unique canonical vertices from every element
    # > The canonical vertex is the minimum of the node and its periodic counterparts
    canonicalSet = {nodeFirst.get(int(node), int(node)) for elem in elems
                                                        for node in cast(np.ndarray, elem.nodes)[:cast(int, elem.type) % 10]}

    # Create a mapping from each canonical vertex to a unique index
    # > FEMVertexID starts at 1
    sortedCanonical = sorted(canonicalSet)
    FEMNodeMapping  = { canonical: newID for newID, canonical in enumerate(sortedCanonical, start=1)}

    # EDGE1: Build Per-BC DIRECTED Mappings
    # > To create a unique signature for each edge, we need to know how its nodes
    # > are displaced by each periodic boundary condition. We build a simple, directed
    # > map for each BC that directly reflects the (source -> target) relationship in
    # > periNodes. We only want to map from negative to positive, thus keep the direction
    periNames = sorted(list({bc for _, bc in periNodes.keys()}))

    # This dictionary holds the directed mapping for each BC
    # > Key: Source node
    # > Val: Target node
    nodeMapBC: dict[str, dict[int, int]] = {bc: {int(n): int(p) for (n, b), p in periNodes.items() if b == bc} for bc in periNames}
    # For convenience, also build a set of all nodes that lie on any boundary

    BCNodes = {int(node)      for node, _ in periNodes.keys()}     # noqa: E272
    BCNodes.update({int(peri) for peri    in periNodes.values()})  # noqa: E272

    # EDGE2: Enumerate All Raw Edges from the Mesh
    # > tuple(element_index, local_edge_index, (node0, node1))
    edgesRaw = []
    for elemID, elem in enumerate(elems):
        elemType  = elem.type
        elemNodes = cast(np.ndarray, elem.nodes)
        for edge in edges(elemType):
            # Get the local corner indices for the current edge
            corners = edge_to_corner(edge, elemType)
            # Get the global node indices for those corners
            n0, n1 = int(elemNodes[corners[0]]), int(elemNodes[corners[1]])
            edgesRaw.append((elemID, edge, (n0, n1)))

    # EDGE3: Generate Canonical Edge Keys (Graph-Based Approach)
    # > This identifies all periodically equivalent edges by building a graph of their representations
    # > 1. Build an Equivalence Graph
    #      The keys are edge-representations (tuples), and the values are sets of other edge-representations they are connected to
    edgeGraph = defaultdict(set)

    for _, _, nodes in edgesRaw:
        # Start with the edge's own representation
        n0, n1   = nodes
        # Find all possible displaced representations
        edgeBase = (n0, n1) if n0 < n1 else (n1, n0)
        edgePeri = {edgeBase}

        # Check if all nodes are on a boundary
        if n0 in BCNodes and n1 in BCNodes:
            for bc in periNames:
                nodeMap = nodeMapBC[bc]
                if n0 in nodeMap and n1 in nodeMap:
                    p0, p1 = nodeMap[n0], nodeMap[n1]
                    if p0 != p1:
                        edgePeri.add((p0, p1) if p0 < p1 else (p1, p0))

        # Connect all representations for this edge in the graph
        # > This ensures that if A->B and B->C, we can later find that A is related to C
        edgeList = list(edgePeri)
        for i in range(len(edgeList)):
            for j in range(i + 1, len(edgeList)):
                u, v = edgeList[i], edgeList[j]
                edgeGraph[u].add(v)
                edgeGraph[v].add(u)

    # > 2. Find connected components and their canonical representatives
    #      This dict stores the single "canonical edge" for each group
    edgeCanonical = {}
    visited       = set()

    # Iterate through all unique representations
    edgeSet = set(edgeGraph.keys())
    for _, _, nodes in edgesRaw:
        n0, n1 = nodes
        edgeSet.add((n0, n1) if n0 < n1 else (n1, n0))

    for nodeStart in edgeSet:
        # Done with this node
        if nodeStart in visited:
            continue

        # This component holds all edge representations for one unique FEM edge
        component = set()
        stack     = [nodeStart]

        while stack:
            currentNode = stack.pop()
            if currentNode in visited:
                continue

            visited  .add(currentNode)
            component.add(currentNode)

            for neighbor in edgeGraph.get(currentNode, []):
                if neighbor not in visited:
                    stack.append(neighbor)

        # > 3. The true canonical key is the minimum representation in the component
        canonical_rep = min(component)

        # > 4. Map all members of the component to this single canonical rep
        for node in component:
            edgeCanonical[node] = canonical_rep

    # > 5. Generate final edge keys for all raw edges
    edgeKeys = []
    for elemID, locEdge, nodes in edgesRaw:
        # Get the initial representation of the edge
        n0, n1   = nodes
        edgeBase = (n0, n1) if n0 < n1 else (n1, n0)

        # Find the true canonical representation from the pre-computed map
        canonical_edge_nodes = edgeCanonical.get(edgeBase, edgeBase)

        v0 = FEMNodeMapping[nodeFirst.get(n0, n0)]
        v1 = FEMNodeMapping[nodeFirst.get(n1, n1)]

        # The final, unique key for this edge
        # > Create the edge key list
        edgeKeys.append((elemID, locEdge, ((v0, v1) if v0 < v1 else (v1, v0), canonical_edge_nodes), (v0, v1), nodes))

    # Create the final mapping from the unique edge key to a simple integer ID
    uniqueEdges    = sorted(list({k for _, _, k, _, _ in edgeKeys}))
    FEMEdgeMapping = {key: i for i, key in enumerate(uniqueEdges)}

    # Build the vertex connectivity
    for elemID, elem in enumerate(elems):
        vertexInfo: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        for locNode, node in enumerate(int(n) for n in cast(np.ndarray, elem.nodes)[:cast(int, elem.type) % 10]):
            # Determine canonical vertex id
            FEMVertexID = FEMNodeMapping[nodeFirst.get(node, node)]
            # Retrive connectivity set for the node
            vertexInfo[locNode] = (FEMVertexID, tuple(sorted(nodeConn.get(node, set()))))
        # Set the vertex connectivity for the element
        elem.vertexInfo = vertexInfo

        # Initialize edgeInfo dictionaries for all elements first
        elem.edgeInfo = {}

    # Use the pre-computed edge_key_list
    for elemID, locEdge, edgeKey, edgePair, edgeNodes in edgeKeys:
        # Retrieve the edge connectivity for the element
        # The structure is: (local_idx, global_id, global_vertex_ids, local_node_ids)
        elems[elemID].edgeInfo[locEdge] = (locEdge, FEMEdgeMapping[edgeKey], edgePair, edgeNodes)


def getFEMInfo(nodeInfo: npt.NDArray) -> tuple[npt.NDArray,  # FEMElemInfo
                                               int,          # nVertices
                                               npt.NDArray,  # VertexInfo
                                               npt.NDArray,  # VertexConnectInfo
                                               int,          # nEdges
                                               npt.NDArray,  # EdgeInfo
                                               npt.NDArray   # EdgeConnectInfo
                                              ]:
    """ Extract the FEM connectivity information and return five arrays

     - FEMElemInfo      : [offsetIndEdge, lastIndEdge, offsetIndVertex, lastIndVertex]
     - VertexInfo       : [FEMVertexID, offsetIndVertexConnect, lastIndVertexConnect]
     - VertexConnectInfo: [nbElemId, nbLocVertexId]
     - EdgeInfo         : [FEMEdgeID,   offsetIndEdgeConnect,   lastIndEdgeConnect]
     - EdgeConnectInfo  : [nbElemID, nbLocEdgeID]
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------

    elems  = mesh_vars.elems
    nElems = len(elems)

    # Check if elements contain FEM connectivity
    if not hasattr(elems[0], 'vertexInfo') or elems[0].vertexInfo is None:
        return np.array([]), 0, np.array([]), np.array([]), 0, np.array([]), np.array([])

    # Vertex connectivity info ---------------------------------------------------
    # > Build list of all vertex occurrences, appearing in the same order as the elements
    occList = tuple((FEMVertexID, elemID, locNode) for elemID , elem             in enumerate(elems)  # noqa: E272
                                                   for locNode, (FEMVertexID, _) in elem.vertexInfo.items())
    nFEMVertices = max(FEMVertexID for FEMVertexID, _, _ in occList)

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
        vertInfo = cast(dict, elem.vertexInfo)
        for _ in vertInfo:
            # Get the occurrence information from the global occList
            FEMVertexID, _, locNode = occList[occGlobalIdx]
            groupOcc = groups[FEMVertexID]
            offset   = len(vertexConnList)

            # Identify the master occurrence (lowest occIdx from the occurrence group)
            masterOcc = min(x[0] for x in groupOcc)

            # Build connectivity list for current element, excluding itself
            connections = [(nbElem+1, nbLocal+1) if   otherOcc == masterOcc else (-(nbElem+1), nbLocal+1)  # noqa: E271
                                                 for (otherOcc, nbElem, nbLocal) in groupOcc if otherOcc != occGlobalIdx]

            # Append vertex information
            vertexConnList.extend(connections)
            vertexInfoList.append([FEMVertexID, offset, offset + len(connections)])
            occGlobalIdx += 1

        # Set the vertex connectivity offset for this element.
        nVertInfo              = len(vertInfo)
        FEMElemInfo[elemID, 2] = vertexOffset
        FEMElemInfo[elemID, 3] = vertexOffset + nVertInfo
        vertexOffset          += nVertInfo

    # Edge   connectivity info ---------------------------------------------------
    # > Build list of all raw edge occurrences, appearing in the same order as the elements
    occList = tuple((edgeIdx, elemID, locEdge, edgeNodes) for elemID, elem                        in enumerate(elems)        # noqa: E272, E501
                                                          for locEdge, (_, edgeIdx, _, edgeNodes) in elem.edgeInfo.items())  # noqa: E501
    # > EdgeID starts at zero, so add 1
    nFEMEdges = max(d[0] for d in occList) + 1

    # 2. Group these occurrences by their canonical FEMEdgeID
    groups = defaultdict(list)
    for occ in occList:
        groups[occ[0]].append(occ)

    edgeInfoList   = []  # List: [FEMEdgeID, offsetIndEdgeConnect, lastIndEdgeConnect]
    edgeConnList   = []  # List: [[nbElemID, nbLocEdgeID]]
    edgeOffset     = 0
    occGlobalIdx   = 0   # global index in occList

    for elemID, elem in enumerate(elems):
        # Process edge occurrences for the current element
        edgeInfo = cast(dict, elem.edgeInfo)
        for _ in range(len(edgeInfo)):
            # Get the occurrence information from the global occList
            currentEdge = occList[occGlobalIdx]
            FEMEdgeID, curElem, curLoc, _ = currentEdge

            # Get all siblings (including periodic ones) from the edge group
            groupOcc = groups[FEMEdgeID]
            offset   = len(edgeConnList)

            # Identify the master occurrence (lowest occIdx from the occurrence group)
            # WARNING: The master/slave logic is simplified here. We assume the first occurrence in the list is the master
            masterOcc = groupOcc[0]

            # Build connectivity list for current element, excluding itself
            connections = []
            for sibling in groupOcc:
                sibEdgeID, sibElem, sibLoc, sibNodes = sibling
                if sibElem == curElem and sibLoc == curLoc:
                    continue

                edgeIsMaster  = (sibElem == masterOcc[1] and sibLoc == masterOcc[2])
                # TODO: Check if the orientation of the master edge is with ascending nodeInfo index
                orientation   = 1 if nodeInfo[masterOcc[3][0]] < nodeInfo[masterOcc[3][1]] else -1

                # The current edge is the master
                if edgeIsMaster:
                    orientedElemID  = -(sibElem + 1)
                    orientedLocEdge =   sibLoc  + 1

                # Current edge is a slave edge
                else:
                    orientation     = orientation if nodeInfo[sibNodes[0]] == nodeInfo[masterOcc[3][0]] else -1
                    orientedElemID  =  sibElem + 1
                    orientedLocEdge =  sibLoc  + 1

                # TODO: Check if this is correct
                # > Copy the orientation
                connections.append([orientedElemID, int(orientedLocEdge * orientation)])

            # Append edge information
            edgeConnList.extend(connections)
            edgeInfoList.append([FEMEdgeID, offset, offset + len(connections)])
            occGlobalIdx += 1

        # Set the edge connectivity offset for this element
        nEdgeInfo              = len(edgeInfo)
        FEMElemInfo[elemID, 0] = edgeOffset
        FEMElemInfo[elemID, 1] = edgeOffset + nEdgeInfo
        edgeOffset            += nEdgeInfo

    # Convert lists to numpy arrays
    vertexInfo = np.array(vertexInfoList, dtype=np.int32)
    vertexConn = np.array(vertexConnList, dtype=np.int32) if vertexConnList else np.array((0, 2), dtype=np.int32)

    edgeInfo   = np.array(edgeInfoList  , dtype=np.int32)
    edgeConn   = np.array(edgeConnList  , dtype=np.int32) if edgeConnList   else np.array((0, 2), dtype=np.int32)  # noqa: E272

    return FEMElemInfo, nFEMVertices, vertexInfo, vertexConn, nFEMEdges, edgeInfo, edgeConn
