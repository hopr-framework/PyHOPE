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
# import sys
from collections import defaultdict
from itertools import chain
from typing import Dict, Tuple
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


def FEMConnect() -> None:
    """ Generate connectivity information for edges and vertices
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import CountOption, GetLogical
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
        for node in (int(n) for n in elem.nodes):
            nodeToElements[node].add(idx)

    # Precompute combined connectivity for each node
    # > For a given node, the combined set is:
    # > nodeToElements[node] ∪ nodeToElements[periDict[node]]
    nodeConn = { node: elemSet.union(nodeToElements.get(periDict.get(node), set()))
                 for node, elemSet in nodeToElements.items()}

    # Collect all unique canonical vertices from every element
    # > The canonical vertex is the minimum of the node and its periodic counterpart
    canonicalSet = { min(node, periDict.get(node, node)) for elem in elems
                                                         for node in map(int, elem.nodes[:elem.type % 100])}

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

    # for idx, elem in enumerate(elems):
    #     print(f'Element {idx}: {elem.nodes} -> {elem.vertexInfo}')

    # sys.exit()


def getFEMInfo() -> tuple[np.ndarray,  # FEMElemInfo
                          np.ndarray,  # VertexInfo
                          np.ndarray   # VertexConnectInfo
                         ]:
    """ Extract the FEM connectivity information and return four arrays

     - FEMElemInfo      : [offsetIndEdge, lastIndEdge, offsetIndVertex, lastIndVertex]
     - vertexInfo       : [FEMVertexID, offsetIndVertexConnect, lastIndVertexConnect]
     - vertexConnectInfo: [nbElemId, nbLocVertexId]
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------

    elems  = mesh_vars.elems
    nElems = len(elems)

    # Check if elements contain FEM connectivity
    if not hasattr(elems[0], 'vertexInfo') or elems[0].vertexInfo is None:
        return np.array([]), np.array([]), np.array([])

    # Vertex connectivity information
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
    vertexConnList = []  # List: [[nbElemId, nbLocVertexId]]

    offset         = 0
    elemOffset     = 0
    occGlobalIdx   = 0   # global index in occList

    for elemID, elem in enumerate(elems):
        # Process vertex occurrences for the current element
        for _ in elem.vertexInfo:
            # Get the occurrence information from the global occList
            FEMVertexId, _, locNode = occList[occGlobalIdx]
            groupOcc = groups[FEMVertexId]
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
            vertexInfoList.append([FEMVertexId, offset, lastIndex])
            occGlobalIdx += 1

        # Set the vertex connectivity offset for this element.
        FEMElemInfo[elemID, 2] = elemOffset
        FEMElemInfo[elemID, 3] = elemOffset + len(elem.vertexInfo)
        elemOffset += len(elem.vertexInfo)

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

    return FEMElemInfo, vertexInfo, vertexConn
