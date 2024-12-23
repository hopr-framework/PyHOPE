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
import copy
import heapq
import itertools
import sys
import traceback
from typing import Optional, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.output.output as hopout
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def flip_analytic(side: list[int], nbside: list[int]) -> int:
    """ Determines the flip of the side-to-side connection based on the analytic side ID
        flip = 1 : 1st node of neighbor side = 1st node of side
        flip = 2 : 2nd node of neighbor side = 1st node of side
        flip = 3 : 3rd node of neighbor side = 1st node of side
        flip = 4 : 4th node of neighbor side = 1st node of side
    """
    # Local imports ----------------------------------------
    from pyhope.common.common import find_index
    # ------------------------------------------------------
    return find_index(nbside, side[0])


def flip_physical(corners: np.ndarray, nbcorners: np.ndarray, tol: float, msg: str) -> int:
    """ Determines the flip of the side-to-side connection based on the physical positions
        flip = 1 : 1st node of neighbor side = 1st node of side
        flip = 2 : 2nd node of neighbor side = 1st node of side
        flip = 3 : 3rd node of neighbor side = 1st node of side
        flip = 4 : 4th node of neighbor side = 1st node of side
    """
    ptree     = spatial.KDTree(nbcorners)

    trCorn    = ptree.query(corners)
    if trCorn[0] > tol:
        hopout.warning(f'Could not determine flip of {msg} side within tolerance {tol}, exiting...')
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    flipID = cast(int, trCorn[1]) + 1
    return flipID


def connect_sides(sideIDs: list[int], sides: list, flipID: int) -> None:
    """ Connect the master and slave sides
    """
    # sides[sideIDs[0]].update(
    #     # Master side contains positive global side ID
    #     MS         = 1,                         # noqa: E251
    #     connection = sideIDs[1],                # noqa: E251
    #     flip       = flipID,                    # noqa: E251
    #     nbLocSide  = sides[sideIDs[1]].locSide  # noqa: E251
    # )
    # sides[sideIDs[1]].update(
    #     MS         = 0,                         # noqa: E251
    #     connection = sideIDs[0],                # noqa: E251
    #     flip       = flipID,                    # noqa: E251
    #     nbLocSide  = sides[sideIDs[0]].locSide  # noqa: E251
    # )
    sides[sideIDs[0]].MS         = 1                          # noqa: E251
    sides[sideIDs[0]].connection = sideIDs[1]                 # noqa: E251
    sides[sideIDs[0]].flip       = flipID                     # noqa: E251
    sides[sideIDs[0]].nbLocSide  = sides[sideIDs[1]].locSide  # noqa: E251
    sides[sideIDs[1]].MS         = 0                          # noqa: E251
    sides[sideIDs[1]].connection = sideIDs[0]                 # noqa: E251
    sides[sideIDs[1]].flip       = flipID                     # noqa: E251
    sides[sideIDs[1]].nbLocSide  = sides[sideIDs[0]].locSide  # noqa: E251


def connect_mortar_sides(sideIDs: list, elems: list, sides: list, nConnSide: list) -> list:
    """ Connect the master (big mortar) and the slave (small mortar) sides
        > Create the virtual sides as needed
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_vars import SIDE
    from pyhope.mesh.mesh_common import type_to_mortar_flip
    # ------------------------------------------------------

    # Get the master and slave sides
    masterSide    = sides[sideIDs[0]]
    masterElem    = elems[masterSide.elemID]
    # masterType    = masterElem['Type']
    masterCorners = masterSide.corners

    # Build mortar type and orientation
    nMortars = len(sideIDs[1])
    match nMortars:
        case 2:
            # Check which edges of big and small side are identical to determine the mortar type
            slaveSide    = sides[sideIDs[1][0]]
            slaveCorners = slaveSide.corners

            if   all(s in slaveCorners for s in [masterCorners[0], masterCorners[1]]) or \
                 all(s in slaveCorners for s in [masterCorners[2], masterCorners[3]]):  # noqa: E271
                mortarType = 2
            elif all(s in slaveCorners for s in [masterCorners[1], masterCorners[2]]) or \
                 all(s in slaveCorners for s in [masterCorners[0], masterCorners[3]]):
                mortarType = 3
            else:
                hopout.warning('Could not determine mortar type, exiting...')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

            del slaveSide
            del slaveCorners

            # Sort the small sides
            # > Order according to master corners, [0, 2]
            slaveSides = [ sides[sideID] for i in [0, 2      ]
                                         for sideID in sideIDs[1] if masterCorners[i] in sides[sideID].corners]

        case 4:
            mortarType = 1
            # Sort the small sides
            # > Order according to master corners, [0, 1, 3, 2]
            slaveSides = [ sides[sideID] for i in [0, 1, 3, 2]
                                         for sideID in sideIDs[1] if masterCorners[i] in sides[sideID].corners]

        case _:
            hopout.warning('Found invalid number of sides for mortar side, exiting...')
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

    # Update the master side
    # sides[sideIDs[0]].update(
    #     # Master side contains positive global side ID
    #     MS          = 1,            # noqa: E251
    #     connection  = -mortarType,  # noqa: E251
    #     flip        = 0,            # noqa: E251
    #     nbLocSide   = 0,            # noqa: E251
    # )
    sides[sideIDs[0]].MS          = 1            # noqa: E251
    sides[sideIDs[0]].connection  = -mortarType  # noqa: E251
    sides[sideIDs[0]].flip        = 0            # noqa: E251
    sides[sideIDs[0]].nbLocSide   = 0            # noqa: E251

    # Update the elems
    for elem in elems:
        for key, val in enumerate(elem.sides):
            # Update the sideIDs
            if val > masterSide.sideID:
                sides[val].sideID += nMortars
                elem.sides[key]   += nMortars
            # Update the connections
            if sides[val].connection is not None and sides[val].connection > masterSide.sideID:
                sides[val].connection += nMortars

    flipMap = type_to_mortar_flip(mesh_vars.elems[masterSide.elemID].type)

    match mortarType:
        case 1:  # 4-1 mortar
            mortarCorners = [0, 1, 3, 2]  # Prepare for non-quad mortars
        case 2:  # 2-1 mortar, split in eta
            mortarCorners = [0, 3]  # Prepare for non-quad mortars
        case 3:  # 2-1 mortar, split in xi
            mortarCorners = [0, 2]  # Prepare for non-quad mortars

    # Insert the virtual sides
    for key, val in enumerate(slaveSides):
        tol        = mesh_vars.tolInternal
        points     = mesh_vars.mesh.points[masterSide.corners]
        nbcorners  = mesh_vars.mesh.points[val.corners]

        flipID = flip_physical(points[mortarCorners[key]], nbcorners, tol, 'mortar')
        flipID = flipMap.get(mortarCorners[key], {}).get(flipID, flipID)
        # val.update(flip=flipID)
        val.flip = flipID

        # Insert the virtual sides
        side = SIDE(sideType   = 104,                          # noqa: E251
                    elemID     = masterElem.elemID,            # noqa: E251
                    sideID     = masterSide.sideID + key + 1,  # noqa: E251
                    locSide    = masterSide.locSide,           # noqa: E251
                    locMortar  = key + 1,                      # noqa: E251
                    # Virtual sides are always master sides
                    MS         = 1,                            # noqa: E251
                    flip       = flipID,                       # noqa: E251
                    connection = val.sideID,                   # noqa: E251
                    nbLocSide  = val.locSide                   # noqa: E251
                   )

        sides.insert(masterSide.sideID + key + 1, side)
        elems[masterElem.elemID].sides.insert(masterSide.locSide + key, side.sideID)

        # Connect the small (slave) sides to the master side
        # val.update(connection = masterSide.sideID,             # noqa: E251
        #            # Small sides are always slave sides
        #            sideType   = -104,                          # noqa: E251
        #            MS         = 0,                             # noqa: E251
        #            flip       = flipID,                        # noqa: E251
        #           )
        val.connection = masterSide.sideID             # noqa: E251
        val.sideType   = -104                          # noqa: E251
        val.MS         = 0                             # noqa: E251
        val.flip       = flipID                        # noqa: E251

        for s in nConnSide:
            if s.sideID == val.sideID:
                nConnSide.remove(s)
                break

    return nConnSide


def find_bc_index(bcs: list, key: str) -> Optional[int]:
    """ Find the index of a BC from its name in the list of BCs
    """
    for iBC, bc in enumerate(bcs):
        if key in bc.name:
            return iBC
        # Try again without the leading 'BC_'
        if key[:3] == 'BC_' and key[3:] in bc.name:
            return iBC
    return None


def find_closest_side(points: np.ndarray, stree: spatial.KDTree, tol: float, msg: str, doMortars: bool = False) -> int:
    """ Query the tree for the closest side
    """
    trSide = stree.query(points)

    # Check if the found side is within tolerance
    # trSide contains the Euclidean distance and the index of the
    # opposing side in the nbFaceSet
    if trSide[0] > tol:
        # Mortar sides are allowed to be not connected
        if doMortars:
            return -1

        hopout.warning(f'Could not find {msg} side within tolerance {tol}, exiting...')
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    return cast(int, trSide[1])


def find_mortar_match(targetCorners: np.ndarray, comboSides: list, mesh: meshio.Mesh, tol: float) -> bool:
    """ Check if the combined points of candidate sides match the target side within tolerance.
    """
    targetPoints = mesh.points[targetCorners]
    ttree = spatial.KDTree(targetPoints)

    comboCorners = [s.corners for s in comboSides]
    comboPoints  = np.concatenate([mesh.points[c] for c in comboCorners], axis=0)
    distances, indices = ttree.query(comboPoints)

    # At least one combo point must match each target point
    matchedIndices = np.unique(indices[distances <= tol])
    if len(matchedIndices) < 4:
        return False

    # Check if exactly one combo point matches each target point
    for point in targetPoints:
        # if not np.allclose(comboPoints, point, atol=tol, rtol=0):
        if np.sum(np.linalg.norm(comboPoints - point, axis=1) <= tol) != 1:
            return False

    # Build the target edges
    targetEdges = build_edges(targetCorners, mesh)
    matches     = []

    # First, check for 2-1 matches
    if len(comboSides) == 2:
        sideEdges = [build_edges(side.corners, mesh) for side in comboSides]

        # Look for 2-1 matches, we need exactly one common edge
        for edge in sideEdges[0]:
            targetP    = edge[:2]  # Start and end points (iX, jX)
            targetDist = edge[2]   # Distance between points

            # Initialize a list to store the matching combo edges for the current target edge
            matchEdges = []

            for comboEdge in sideEdges[1]:
                comboP    = comboEdge[:2]  # Start and end points (iX, jX)
                comboDist = comboEdge[2]   # Distance between points

                # Check if the points match and the distance is the same, taking into account the direction
                if (all(np.isclose(tp, cp) for tp, cp in zip(targetP, comboP)) or
                    all(np.isclose(tp, cp) for tp, cp in zip(targetP, reversed(comboP)))) and \
                    np.isclose(targetDist, comboDist):
                    matchEdges.append(comboEdge)

            # This should result in exactly 1 match
            if len(matchEdges) == 1:
                matches.append((edge, matchEdges.pop()))

        # We only allow 2-1 matches, so in the end we should have exactly 1 match
        if len(matches) != 1:
            return False

        # Here, we only allow 2-1 matches
        comboEdges  = [e for s in comboSides for e in build_edges(s.corners, mesh)]
        comboEdges  = find_edge_combinations(comboEdges)

        # Attempt to match the target edges with the candidate edges
        matches     = []  # List to store matching edges

        # Iterate over each target edge
        for targetEdge in targetEdges:
            targetP    = targetEdge[:2]  # Start and end points (iX, jX)
            targetDist = targetEdge[2]   # Distance between points

            # Initialize a list to store the matching combo edges for the current target edge
            matchEdges = []

            # Iterate over comboEdges to find matching edges
            for comboEdge in comboEdges:
                comboP    = comboEdge[:2]  # Start and end points (iX, jX)
                comboDist = comboEdge[2]   # Distance between points

                # Check if the points match and the distance is the same, taking into account the direction
                if (all(np.isclose(tp, cp) for tp, cp in zip(targetP, comboP)) or
                    all(np.isclose(tp, cp) for tp, cp in zip(targetP, reversed(comboP)))) and \
                   np.isclose(targetDist, comboDist):
                    matchEdges.append(comboEdge)

            # This should result in exactly 1 match
            if len(matchEdges) > 1:
                return False
            elif len(matchEdges) == 1:
                matches.append((targetEdge, matchEdges.pop()))

        if len(matches) != 2:
            return False

    # Next, check for 4-1 matches
    if len(comboSides) == 4:
        # Check if there is exactly one point that all 4 sides have in common.
        common_points = set(comboSides[0].corners)
        matchFound = False
        for p in common_points:
            # Check if all 4 sides have the point
            matchedPoints = 0
            for side in comboSides[1:]:
                for p1 in side.corners:
                    if (all(np.isclose(tp, cp) for tp, cp in zip(mesh.points[p], mesh.points[p1]))):
                        matchedPoints += 1

            if matchedPoints == 3:
                matchFound = True
                break

        if not matchFound:
            return False

        comboEdges  = [e for s in comboSides for e in build_edges(s.corners, mesh)]
        comboEdges  = find_edge_combinations(comboEdges)

        # Attempt to match the target edges with the candidate edges
        matches     = []  # List to store matching edges

        # Iterate over each target edge
        for targetEdge in targetEdges:
            targetP    = targetEdge[:2]  # Start and end points (iX, jX)
            targetDist = targetEdge[2]   # Distance between points

            # Initialize a list to store the matching combo edges for the current target edge
            matchEdges = []

            # Iterate over comboEdges to find matching edges
            for comboEdge in comboEdges:
                comboP    = comboEdge[:2]  # Start and end points (iX, jX)
                comboDist = comboEdge[2]  # Distance between points

                # Check if the points match and the distance is the same, taking into account the direction
                if (all(np.isclose(tp, cp) for tp, cp in zip(targetP, comboP)) or
                    all(np.isclose(tp, cp) for tp, cp in zip(targetP, reversed(comboP)))) and \
                   np.isclose(targetDist, comboDist):
                    matchEdges.append(comboEdge)

            # This should result in exactly 1 match
            if len(matchEdges) > 1:
                return False
            elif len(matchEdges) == 1:
                matches.append((targetEdge, matchEdges.pop()))

        if len(matches) != 4:
            return False

    # Found a valid match
    return True


def build_edges(corners: np.ndarray, mesh: meshio.Mesh) -> list[tuple]:
    """Build edges from the 4 corners of a quadrilateral, considering CGNS ordering
    """
    edges = [
        (corners[0], corners[1], np.linalg.norm(mesh.points[corners[0]] - mesh.points[corners[1]])),  # Edge between points 0 and 1
        (corners[1], corners[2], np.linalg.norm(mesh.points[corners[1]] - mesh.points[corners[2]])),  # Edge between points 1 and 2
        (corners[2], corners[3], np.linalg.norm(mesh.points[corners[2]] - mesh.points[corners[3]])),  # Edge between points 2 and 3
        (corners[3], corners[0], np.linalg.norm(mesh.points[corners[3]] - mesh.points[corners[0]])),  # Edge between points 3 and 0
    ]
    return edges


def find_edge_combinations(comboEdges):
    """Build combinations of edges that share exactly one point and form a line
    """
    # Local imports ----------------------------------------
    from collections import defaultdict
    from itertools import combinations
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------

    # Create a dictionary to store edges by their shared points
    pointToEdges = defaultdict(list)

    # Fill the dictionary with edges indexed by their points
    for i, j, dist in comboEdges:
        pointToEdges[i].append((i, j, dist))
        pointToEdges[j].append((i, j, dist))

    # Initialize an empty list to store the valid combinations of edges
    validCombo = []

    # Iterate over all points and their associated edges
    for _, edges in pointToEdges.items():
        if len(edges) < 2:  # Skip points with less than 2 edges
            continue

        # Now, we generate all possible pairs of edges that share the point
        for edge1, edge2 in combinations(edges, 2):
            # Ensure the edges are distinct and share exactly one point
            # Since both edges share 'point', they are valid combinations
            # We store the combination as an np.array (i, j, dist)
            i1, j1, _ = edge1
            i2, j2, _ = edge2

            # Use set operations to determine the unique start and end points
            commonPoint = {i1, j1} & {i2, j2}
            if len(commonPoint) == 1:  # Check that there's exactly one shared point
                commonPoint = commonPoint.pop()

                # Exclude the common point and get the unique start and end points
                edgePoints = np.array([i1, j1, i2, j2])

                # Find the index of the common point and delete it
                commonIndex = np.where( edgePoints == commonPoint)[0]
                edgePoints  = np.delete(edgePoints, commonIndex)

                # The remaining points are the start and end points of the edge combination
                point1, point2 = edgePoints

                # Get the coordinates of the points
                p1, p2 = mesh_vars.mesh.points[point1], mesh_vars.mesh.points[point2]
                c1     = mesh_vars.mesh.points[commonPoint]

                # Calculate the bounding box of the two edge points
                bbox_min = np.minimum(p1, p2)
                bbox_max = np.maximum(p1, p2)

                # Check if the common point is within the bounding box of p1 and p2
                if np.all(np.isclose(bbox_min, np.minimum(bbox_min, c1))) and \
                   np.all(np.isclose(bbox_max, np.maximum(bbox_max, c1))):
                    # Calculate the distance between the start and end points
                    lineDist = np.linalg.norm(p1 - p2)

                    # Append the indices and the line distance
                    validCombo.append((point1, point2, lineDist))

    return validCombo


def get_side_id(corners: np.ndarray, side_dict: dict) -> int:
    """ Get sorted corners and hash them to get the side ID
    """
    corners_sorted = np.sort(corners)
    corners_hash = hash(corners_sorted.tobytes())
    return side_dict[corners_hash][0]


def get_nonconnected_sides(sides: list, mesh: meshio.Mesh) -> tuple[list, list[np.ndarray]]:
    """ Get a list of internal sides that are not connected to any
        other side together with a list of their centers
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    # Update the list
    nConnSide = [s for s in sides if s.connection is None and s.bcid is None]
    # Append the inner BCs
    for s in (s for s in sides if s.bcid is not None and s.connection is None):
        if mesh_vars.bcs[s.bcid].type[0] == 0:
            nConnSide.append(s)
    nConnCenter = [np.mean(mesh.points[s.corners], axis=0) for s in nConnSide]
    return nConnSide, nConnCenter


def periodic_update(sideIDs: list[int], vv: dict) -> None:
    """Update the mesh after connecting periodic sides
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_common import face_to_nodes
    from pyhope.mesh.mesh_common import flip_s2m
    # ------------------------------------------------------

    sides    = [mesh_vars.sides[s] for s in sideIDs]
    elems    = [mesh_vars.elems[s.elemID] for s in sides]
    nodes    = np.array([elems[0].nodes[s] for s in face_to_nodes(sides[0].face, elems[0].type, mesh_vars.nGeo)])
    nbNodes  = np.array([elems[1].nodes[s] for s in face_to_nodes(sides[1].face, elems[1].type, mesh_vars.nGeo)])

    # Get the flip map
    indices = flip_s2m(mesh_vars.nGeo+1, 1 if sides[0].flip <= 2 else sides[0].flip)

    # for iy, ix in np.ndindex(nodes.shape[:2]):
    #     node   = nodes[ix, iy]
    #     nbNode = nbNodes[indices[ix, iy, 0], indices[ix, iy, 1]]
    #
    #     # Sanity check if the periodic vector matches
    #     if not np.allclose(vv['Dir'], mesh_vars.mesh.points[nbNode] - mesh_vars.mesh.points[node],
    #                        rtol=mesh_vars.tolPeriodic, atol=mesh_vars.tolPeriodic):
    #         hopout.warning('Error in periodic update, periodic vector does not match!')
    #         sys.exit(1)
    #
    #     # Center between both points
    #     center = 0.5 * (mesh_vars.mesh.points[node] + mesh_vars.mesh.points[nbNode])
    #
    #     lowerP = copy.copy(center)
    #     upperP = copy.copy(center)
    #     for key, val in enumerate(vv['Dir']):
    #         lowerP[key] -= 0.5 * val
    #         upperP[key] += 0.5 * val
    #
    #     mesh_vars.mesh.points[  node] = lowerP
    #     mesh_vars.mesh.points[nbNode] = upperP

    # Extract relevant indices from the mesh
    nbNodes = nbNodes[indices[:, :, 0], indices[:, :, 1]]

    # Check if periodic vector matches using vectorized np.allclose
    if not np.allclose(mesh_vars.mesh.points[nodes] + vv['Dir'], mesh_vars.mesh.points[nbNodes],
                       rtol=mesh_vars.tolPeriodic, atol=mesh_vars.tolPeriodic):
        hopout.warning('Error in periodic update, periodic vector does not match!')
        sys.exit(1)

    # Calculate the center for both points
    centers = 0.5 * (mesh_vars.mesh.points[nodes] + mesh_vars.mesh.points[nbNodes])

    # Update the mesh points for both node and nbNode
    mesh_vars.mesh.points[nodes]   = centers - 0.5*vv['Dir']
    mesh_vars.mesh.points[nbNodes] = centers + 0.5*vv['Dir']


def ConnectMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.common.common import find_index
    from pyhope.common.common_progress import ProgressBar
    from pyhope.io.io_vars import MeshFormat
    from pyhope.readintools.readintools import GetLogical
    from pyhope.mesh.mesh_common import face_to_nodes
    # ------------------------------------------------------

    match io_vars.outputformat:
        case MeshFormat.FORMAT_HDF5:
            pass
        case _:
            return

    hopout.separator()
    hopout.info('CONNECT MESH...')
    hopout.sep()

    mesh_vars.doPeriodicCorrect = GetLogical('doPeriodicCorrect')
    mesh_vars.doMortars         = GetLogical('doMortars')
    doPeriodicCorrect = mesh_vars.doPeriodicCorrect
    doMortars         = mesh_vars.doMortars

    mesh    = mesh_vars.mesh
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides

    # cell_sets contain the face IDs [dim=2]
    # > Offset is calculated with entities from [dim=0, dim=1]
    offsetcs = 0
    for key, value in mesh.cells_dict.items():
        if 'vertex' in key:
            offsetcs += value.shape[0]
        elif 'line' in key:
            offsetcs += value.shape[0]

    # Map sides to BC
    # > Create a dict containing only the face corners
    side_corners = dict()
    for elem in elems:
        for iSide, side in enumerate(elem.sides):
            corners = np.sort(sides[side].corners)
            corners = hash(corners.tobytes())
            side_corners.update({side: corners})

    # Build the reverse dictionary
    corner_side = dict()
    for key, val in side_corners.items():
        if val not in corner_side:
            corner_side[val] = [key]
        else:
            corner_side[val].append(key)

    bar = ProgressBar(value=len(sides), title='│                Processing Sides')

    # Try to connect the inner sides
    ninner = 0
    for (key, val) in corner_side.items():
        match len(val):
            case 1:  # BC side
                continue
            case 2:  # Internal side
                sideIDs   = val
                corners   = sides[sideIDs[0]].corners
                nbcorners = sides[sideIDs[1]].corners
                flipID    = flip_analytic(corners, nbcorners) + 1
                # Connect the sides
                connect_sides(sideIDs, sides, flipID)
                ninner += 1

                # Update the progress bar
                [bar.step() for _ in range(2)]
            case _:  # Zero or more than 2 sides
                hopout.warning('Found internal side with more than two adjacent elements, exiting...')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

    # Set BC and periodic sides
    bcs = mesh_vars.bcs
    vvs = mesh_vars.vvs
    csetMap = []
    for key, cset in mesh.cell_sets.items():
        # Check if the set is a BC
        bcID = find_bc_index(bcs, key)
        if bcID is None:
            hopout.warning(f'Could not find BC {key} in list, exiting...')
            sys.exit(1)

        # Find the mapping to the (N-1)-dim elements
        csetMap = [s for s in range(len(cset)) if cset[s] is not None]

        # Get the list of sides
        for iMap in csetMap:
            iBCsides = np.array(cset[iMap]).astype(int) - offsetcs
            mapFaces = mesh.cells[iMap].data

            # Map the unique quad sides to our non-unique elem sides
            for iSide in iBCsides:
                # Get the quad corner nodes
                # FIXME: HARDCODED FIRST 4 NODES WHICH ARE THE OUTER CORNER NODES FOR QUADS!
                corners = np.sort(np.array(mapFaces[iSide][0:4]))
                corners = hash(corners.tobytes())

                # Boundary faces are unique
                # sideID  = find_key(face_corners, corners)
                sideID = corner_side[corners][0]
                # sides[sideID].update(bcid=bcID)
                sides[sideID].bcid = bcID

                if bcs[bcID].type[0] != 1:
                    bar.step()

    # Try to connect the periodic sides
    for key, cset in mesh.cell_sets.items():
        # Check if the set is a BC
        bcID = find_bc_index(bcs, key)
        if bcID is None:
            hopout.warning(f'Could not find BC {key} in list, exiting...')
            sys.exit(1)

        # Only periodic BCs and only try to connect in positive direction
        if bcs[bcID].type[0] != 1 or bcs[bcID].type[3] < 0:
            continue

        # Get the opposite side
        iVV        = bcs[bcID].type[3] - 1
        nbType     = copy.copy(bcs[bcID].type)
        nbType[3] *= -1
        nbBCID     = find_index([s.type for s in bcs], nbType)
        nbBCName   = bcs[nbBCID].name

        # Collapse all opposing corner nodes into an [:, 12] array
        nbCellSet  = mesh.cell_sets[nbBCName]
        # Find the mapping to the (N-1)-dim elements
        nbcsetMap  = [s for s in range(len(nbCellSet)) if nbCellSet[s] is not None
                      and cast(np.ndarray, nbCellSet[s]).size > 0]

        # FIXME: TODO HYBRID MESHES
        if len(nbcsetMap) > 1:
            print('Hybrid meshes currently not supported')
            sys.exit(1)

        # Get the list of sides
        nbFaceSet  =  np.array(nbCellSet[csetMap[0]]).astype(int)
        nbmapFaces = mesh.cells[csetMap[0]].data
        nbCorners  = [np.array(nbmapFaces[s - offsetcs]) for s in nbFaceSet]
        nbPoints   = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
        nbPoints   = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
        del nbCorners

        # Build a k-dimensional tree of all points on the opposing side
        stree    = spatial.KDTree(nbPoints)
        # Get the list of sides on our side
        iBCsides = np.array(cset[csetMap[0]]).astype(int) - offsetcs

        # Map the unique quad sides to our non-unique elem sides
        for iSide in iBCsides:
            # Get the quad corner nodes
            corners   = np.array(nbmapFaces[iSide])
            points    = copy.copy(mesh.points[corners])

            # Shift the points in periodic direction
            for iPoint in range(points.shape[0]):
                points[iPoint, :] += vvs[iVV]['Dir']
            points    = np.sort(points, axis=0).flatten()

            # Query the try for the opposing side
            tol       = np.linalg.norm(vvs[iVV]['Dir'], ord=2).astype(float) * mesh_vars.tolPeriodic
            nbSideIdx = find_closest_side(points, cast(spatial.KDTree, stree), tol, 'periodic')
            nbiSide   = nbFaceSet[nbSideIdx] - offsetcs

            # Get our and neighbor corner quad nodes
            sideID    = get_side_id(nbmapFaces[iSide  ][0:4], corner_side)
            nbSideID  = get_side_id(nbmapFaces[nbiSide][0:4], corner_side)

            # Build the connection, including flip
            sideIDs   = [sideID, nbSideID]
            points    = mesh.points[sides[sideIDs[0]].corners]
            for iPoint in range(points.shape[0]):
                points[iPoint, :] += vvs[iVV]['Dir']

            # > Find the first neighbor point to determine the flip
            nbcorners = mesh.points[sides[sideIDs[1]].corners]
            flipID    = flip_physical(points[0], nbcorners, tol, 'periodic')

            # Connect the sides
            connect_sides(sideIDs, sides, flipID)

            # Update the sides
            if doPeriodicCorrect:
                periodic_update(sideIDs, vvs[iVV])

                # Update the progress bar
                [bar.step() for _ in range(2)]

    # Non-connected sides without BCID are possible inner sides
    nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)
    nInterZoneConnect      = len(nConnSide)

    # Loop over all sides and try to connect
    iter    = 0
    maxIter = copy.copy(nInterZoneConnect)
    tol     = mesh_vars.tolInternal
    # While maxIter should be enough, this results in non-connected mortar sides. We can append a maximum of 4 virtual sides,
    # so let's set the maxIter to 5 just to be safe.
    while len(nConnSide) > 1 and iter <= maxIter:
        # Ensure the loop exits after checking every side
        iter += 1

        # Remove the first side from the list
        targetSide   = nConnSide  .pop(0)
        targetCenter = nConnCenter.pop(0)

        # Collapse all opposing corner nodes into an [:, 12] array
        nbCorners  = [s.corners for s in nConnSide]
        nbPoints   = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
        nbPoints   = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
        del nbCorners

        # Build a k-dimensional tree of all points on the opposing side
        stree      = spatial.KDTree(nbPoints)
        ctree      = spatial.KDTree(nConnCenter)

        # Map the unique quad sides to our non-unique elem sides
        corners    = targetSide.corners
        points     = np.sort(mesh.points[corners], axis=0).flatten()

        # Query the tree for the opposing side
        nbSideIdx  = find_closest_side(points, cast(spatial.KDTree, stree), tol, 'internal', doMortars)

        # Regular internal side
        if nbSideIdx >= 0:
            nbiSide   = nbSideIdx

            # Get our and neighbor corner quad nodes
            sideID    = get_side_id(targetSide.corners        , corner_side)
            nbSideID  = get_side_id(nConnSide[nbiSide].corners, corner_side)

            # Build the connection, including flip
            sideIDs   = [sideID, nbSideID]
            points    = mesh.points[sides[sideIDs[0]].corners]
            # > Find the first neighbor point to determine the flip
            nbcorners = mesh.points[sides[sideIDs[1]].corners]
            flipID    = flip_physical(points[0], nbcorners, tol, 'internal')

            # Connect the sides
            connect_sides(sideIDs, sides, flipID)

            # Update the list
            nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)

            # Update the progress bar
            [bar.step() for _ in range(2)]

            if not doMortars:
                hopout.warning(f'Could not find internal side within tolerance {tol}, exiting...')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

    # Mortar sides
    if doMortars:
        nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)
        nInterZoneConnect      = len(nConnSide)

        iter    = 0
        maxIter = copy.copy(nInterZoneConnect)
        tol     = mesh_vars.tolInternal

        while len(nConnSide) > 1 and iter <= maxIter:
            # Ensure the loop exits after checking every side
            iter += 1

            # Remove the first side from the list
            targetSide   = nConnSide  .pop(0)
            targetCenter = nConnCenter.pop(0)

            # Collapse all opposing corner nodes into an [:, 12] array
            nbCorners  = [s.corners for s in nConnSide]
            nbPoints   = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
            nbPoints   = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
            del nbCorners

            # Build a k-dimensional tree of all points on the opposing side
            stree      = spatial.KDTree(nbPoints)
            ctree      = spatial.KDTree(nConnCenter)

            # Map the unique quad sides to our non-unique elem sides
            corners    = targetSide.corners
            points     = np.sort(mesh.points[corners], axis=0).flatten()

            # Query the tree for the opposing side
            nbSideIdx  = find_closest_side(points, cast(spatial.KDTree, stree), tol, 'internal', doMortars)

            # Mortar side
            # > Here, we can only attempt to connect big to small mortar sides. Thus, if we encounter a small mortar sides which
            # > generates no match, we simply append the side again at the end and try again. As the loop exists after checking
            # > len(nConnSide), we will check each side once.
            if nbSideIdx >= 0:
                continue

            # Calculate the radius of the convex hull
            targetPoints = mesh.points[targetSide.corners]
            targetMinMax = (targetPoints.min(axis=0), targetPoints.max(axis=0))
            targetRadius = np.linalg.norm(targetMinMax[1] - targetMinMax[0], ord=2) / 2.

            # Find nearby sides to consider as candidate mortar sides
            targetNeighbors = ctree.query_ball_point(targetCenter, targetRadius)
            # Eliminate sides in the same element
            targetNeighbors = [s for s in targetNeighbors if nConnSide[s].elemID != targetSide.elemID]

            # Prepare combinations for 2-to-1 and 4-to-1 mortar matching
            candidate_combinations = []
            if len(targetNeighbors) >= 2:
                candidate_combinations += list(itertools.combinations(targetNeighbors, 2))
            if len(targetNeighbors) >= 4:
                candidate_combinations += list(itertools.combinations(targetNeighbors, 4))

            # Attempt to match the target side with candidate combinations
            matchFound   = False
            comboSides   = []
            for comboIDs in candidate_combinations:
                # Get the candidate sides
                comboSides   = [nConnSide[iSide] for iSide in comboIDs]

                # Found a valid match
                if find_mortar_match(targetSide.corners, comboSides, mesh, tol):
                    matchFound = True
                    break

            if matchFound:
                # Get our and neighbor corner quad nodes
                sideID    =  targetSide.sideID
                nbSideID  = [side.sideID for side in comboSides]

                # Build the connection, including flip
                sideIDs   = [sideID, nbSideID]

                # Connect mortar sides and update the list
                nConnSide   = connect_mortar_sides(sideIDs, elems, sides, nConnSide)
                nConnCenter = [np.mean(mesh.points[s.corners], axis=0) for s in nConnSide]

                # Update the progress bar
                [bar.step() for _ in range(len(nbSideID) + 1)]

            # No connection, attach the side at the end
            else:
                nConnSide  .append(targetSide)
                nConnCenter.append(targetCenter)

    nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)
    if len(nConnSide) > 0:
        hopout.warning('Could not connect {} side{}'.format(len(nConnSide), '' if len(nConnSide) == 1 else 's'))

        for side in nConnSide:
            print(hopout.warn(f'> Element {side.elemID+1}, Side {side.face}, Side {side.sideID+1}'))  # noqa: E501
            elem     = elems[side.elemID]
            nodes    = np.transpose(np.array([elem.nodes[s] for s in face_to_nodes(side.face, elem.type, mesh_vars.nGeo)]))
            nodes    = np.transpose(mesh_vars.mesh.points[nodes]         , axes=(2, 0, 1))
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in nodes[:,  0,  0]) + ']'))
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in nodes[:,  0, -1]) + ']'))
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in nodes[:, -1,  0]) + ']'))
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in nodes[:, -1, -1]) + ']'))
            print()
        sys.exit(1)

    # Close the progress bar
    bar.close()

    hopout.separator()
    if nInterZoneConnect > 0:
        hopout.sep()
        hopout.routine('Connected {} inter-zone faces'.format(nInterZoneConnect))

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

            # Set the negative globalSideID of the slave  side
            # sides[nbSideID].update(globalSideID=-(globalSideID))
            sides[nbSideID].globalSideID = -(globalSideID)

    # Count the sides
    nsides             = len(sides)
    sides_conn         = np.array([s.connection is not None                      for s in sides])  # noqa: E271, E272
    sides_bc           = np.array([s.bcid       is not None                      for s in sides])  # noqa: E271, E272
    sides_mortar_big   = np.array([s.connection is not None and s.connection < 0 for s in sides])  # noqa: E271, E272
    sides_mortar_small = np.array([s.locMortar  is not None                      for s in sides])  # noqa: E271, E272

    # Count each type of side
    ninnersides        = np.sum( sides_conn & ~sides_bc & ~sides_mortar_small & ~sides_mortar_big)
    nperiodicsides     = np.sum( sides_conn &  sides_bc & ~sides_mortar_small & ~sides_mortar_big)
    nbcsides           = np.sum(~sides_conn &  sides_bc & ~sides_mortar_small & ~sides_mortar_big)
    nmortarbigsides    = np.sum(                                                 sides_mortar_big)
    nmortarsmallsides  = np.sum(                           sides_mortar_small                    )
    nsides             = len(sides) - nmortarsmallsides

    hopout.sep()
    hopout.info(' Number of sides                : {:12d}'.format(nsides))
    hopout.info(' Number of inner sides          : {:12d}'.format(ninnersides))
    hopout.info(' Number of mortar sides (big)   : {:12d}'.format(nmortarbigsides))
    hopout.info(' Number of mortar sides (small) : {:12d}'.format(nmortarsmallsides))
    hopout.info(' Number of boundary sides       : {:12d}'.format(nbcsides))
    hopout.info(' Number of periodic sides       : {:12d}'.format(nperiodicsides))
    hopout.sep()

    hopout.info('CONNECT MESH DONE!')
