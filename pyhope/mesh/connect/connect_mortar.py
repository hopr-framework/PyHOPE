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
import itertools
import sys
import traceback
from collections import defaultdict
from functools import lru_cache
# from functools import cache
from itertools import combinations
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


def ConnectMortar( nConnSide   : list
                 , nConnCenter: list
                 , mesh       : meshio.Mesh
                 , elems      : list
                 , sides      : list
                 , bar
                 , doPeriodic : bool = False) -> None:
    """ Function to connect mortar sides

        Args:
            doPeriodic: Flag to enable periodic connections
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.connect.connect import find_closest_side
    # ------------------------------------------------------

    # Set BC and periodic sides
    bcs = mesh_vars.bcs
    vvs = mesh_vars.vvs

    nInterConnConnect = len(nConnSide)
    iter    = 0
    maxIter = copy.copy(nInterConnConnect)
    tol     = mesh_vars.tolPeriodic

    while len(nConnSide) > 1 and iter <= maxIter:
        # Ensure the loop exits after checking every side
        iter += 1

        # Remove the first side from the list
        targetSide   = nConnSide  .pop(0)
        targetCenter = nConnCenter.pop(0)

        # Get the opposite side
        if doPeriodic:
            bcID   = targetSide.bcid
            iVV    = bcs[bcID].type[3]
            VV     = vvs[np.abs(iVV)-1]['Dir'] * np.sign(iVV)
        else:
            VV     = None

        # Collapse all opposing corner nodes into an [:, 12] array
        nbCorners  = [s.corners for s in nConnSide]
        nbPoints   = np.sort(mesh.points[nbCorners], axis=1).reshape(len(nbCorners), -1)
        del nbCorners

        # Build a k-dimensional tree of all points on the opposing side
        stree      = spatial.KDTree(nbPoints)
        ctree      = spatial.KDTree(nConnCenter)

        # Map the unique quad sides to our non-unique elem sides
        corners    = targetSide.corners

        if doPeriodic:
            # Shift the points in periodic direction
            points     = mesh.points[corners].copy()
            points    += VV
            points     = np.sort(points, axis=0).flatten()
            targetCenter += VV
        else:
            points     = np.sort(mesh.points[corners], axis=0).flatten()

        # Query the tree for the opposing side
        nbSideIdx  = find_closest_side(points, cast(spatial.KDTree, stree), tol, 'internal', doMortars=True)

        # Mortar side
        # > Here, we can only attempt to connect big to small mortar sides. Thus, if we encounter a small mortar sides which
        # > generates no match, we simply append the side again at the end and try again. As the loop exists after checking
        # > len(nConnSide), we will check each side once.
        if nbSideIdx >= 0:
            continue

        # Calculate the radius of the convex hull
        targetPoints = mesh.points[corners].copy()
        targetMinMax = (targetPoints.min(axis=0), targetPoints.max(axis=0))
        targetRadius = np.linalg.norm(targetMinMax[1] - targetMinMax[0], ord=2) / 2.

        # Find nearby sides to consider as candidate mortar sides
        # > Eliminate sides in the same element
        targetNeighbors = [s for s in ctree.query_ball_point(targetCenter, targetRadius) if nConnSide[s].elemID != targetSide.elemID]  # noqa: E501

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
            if find_mortar_match(targetSide.corners, comboSides, mesh, tol, VV):
                matchFound = True
                break

        if matchFound:
            # Get our and neighbor corner quad nodes
            sideID    =  targetSide.sideID
            nbSideID  = [side.sideID for side in comboSides]

            # Build the connection, including flip
            sideIDs   = [sideID, nbSideID]

            # Connect mortar sides and update the list
            nConnSide   = connect_mortar_sides(sideIDs, elems, sides, nConnSide, VV)
            nConnCenter = [np.mean(mesh.points[s.corners], axis=0) for s in nConnSide]

            # Update the progress bar
            bar.step(len(nbSideID) + 1)

        # No connection, attach the side at the end
        else:
            nConnSide  .append(targetSide)
            nConnCenter.append(targetCenter)


def points_exist_in_target(pts, slavePts, tol) -> bool:
    """ Check if the combined points of candidate sides match the target side within tolerance.
    """
    return all(any(np.allclose(pt, sp, rtol=tol, atol=tol) for sp in slavePts) for pt in pts)


def point_exists_in_side(masterPoint, slaveSide, tol) -> bool:
    """ Check if the combined points of candidate sides match the target side within tolerance.
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    slavePoints = mesh_vars.mesh.points[slaveSide.corners]
    return any(np.allclose(masterPoint, sp, rtol=tol, atol=tol) for sp in slavePoints)


def connect_mortar_sides( sideIDs  : list
                        , elems    : list
                        , sides    : list
                        , nConnSide: list
                        , vv       : Optional[np.ndarray] = None) -> list:
    """ Connect the master (big mortar) and the slave (small mortar) sides
        > Create the virtual sides as needed
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.connect.connect import flip_physical
    from pyhope.mesh.mesh_vars import SIDE
    from pyhope.mesh.mesh_common import type_to_mortar_flip
    # ------------------------------------------------------

    # Get the master and slave sides
    masterSide    = sides[sideIDs[0]]
    masterElem    = elems[masterSide.elemID]
    # masterType    = masterElem['Type']
    masterCorners = masterSide.corners
    masterPoints  = mesh_vars.mesh.points[masterCorners].copy()

    if vv is not None:
        # Shift the points in periodic direction
        masterPoints += vv
    tol = mesh_vars.tolInternal if vv is None else mesh_vars.tolPeriodic

    # Build mortar type and orientation
    nMortars = len(sideIDs[1])
    match nMortars:
        case 2:
            # Check which edges of big and small side are identical to determine the mortar type
            slaveSide    = sides[sideIDs[1][0]]
            slaveCorners = slaveSide.corners
            slavePoints  = mesh_vars.mesh.points[slaveCorners]

            # Check which edges match
            if   points_exist_in_target([masterPoints[0], masterPoints[1]], slavePoints, tol) or \
                 points_exist_in_target([masterPoints[2], masterPoints[3]], slavePoints, tol):  # noqa: E271
                mortarType = 2
            elif points_exist_in_target([masterPoints[1], masterPoints[2]], slavePoints, tol) or \
                 points_exist_in_target([masterPoints[0], masterPoints[3]], slavePoints, tol):
                mortarType = 3
            else:
                hopout.warning('Could not determine mortar type, exiting...')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

            del slaveSide
            del slaveCorners

            # Sort the small sides
            # > Order according to master corners, [0, 2]
            # slaveSides = [ sides[sideID] for i in [0, 2      ]
            #                              for sideID in sideIDs[1] if masterCorners[i] in sides[sideID].corners]
            slaveSides = [ sides[sideID] for i in [0, 2]
                                         for sideID in sideIDs[1] if point_exists_in_side(masterPoints[i], sides[sideID], tol)
        ]

        case 4:
            mortarType = 1
            # Sort the small sides
            # > Order according to master corners, [0, 1, 3, 2]
            # slaveSides = [ sides[sideID] for i in [0, 1, 3, 2]
            #                              for sideID in sideIDs[1] if masterCorners[i] in sides[sideID].corners]
            slaveSides = [ sides[sideID] for i in [0, 1, 3, 2]
                                         for sideID in sideIDs[1] if point_exists_in_side(masterPoints[i], sides[sideID], tol)
        ]

        case _:
            hopout.warning('Found invalid number of sides for mortar side, exiting...')
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

    # Sanity check
    if len(slaveSides) != nMortars:
        hopout.warning('Could not determine mortar type, exiting...')
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
        nbcorners  = mesh_vars.mesh.points[val.corners]

        flipID = flip_physical(masterPoints[mortarCorners[key]], nbcorners, tol, 'mortar')
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


def find_mortar_match( targetCorners: np.ndarray
                     , comboSides   : list
                     , mesh         : meshio.Mesh
                     , tol          : float
                     , vv           : Optional[np.ndarray] = None) -> bool:
    """ Check if the combined points of candidate sides match the target side within tolerance.
    """
    targetPoints = mesh.points[targetCorners].copy()
    if vv is not None:
        # Shift the points in periodic direction
        targetPoints += vv

    ttree = spatial.KDTree(targetPoints)

    comboCorners = [s.corners for s in comboSides]
    comboPoints  = np.concatenate([mesh.points[c] for c in comboCorners], axis=0)
    distances, indices = map(np.array, ttree.query(comboPoints))

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
    # INFO: Uncached version
    # targetEdges = build_edges(targetCorners, mesh.points[targetCorners])
    # INFO: Cached version
    targetEdges = build_edges(arrayToTuple(targetCorners), tuple(map(tuple, mesh.points[targetCorners])))
    matches     = []

    # First, check for 2-1 matches
    if len(comboSides) == 2:
        # INFO: Uncached version
        # sideEdges = [build_edges(side.corners, mesh.points[side.corners]) for side in comboSides]
        # INFO: Cached version
        sideEdges = [build_edges(arrayToTuple(side.corners), tuple(map(tuple, mesh.points[side.corners]))) for side in comboSides]

        # Look for 2-1 matches, we need exactly one common edge
        for edge in sideEdges[0]:
            # targetP    = edge[:2]  # Start and end points (iX, jX)
            targetP    = [mesh.points[s] for s in edge[:2]]
            targetDist = edge[2]   # Distance between points

            # Initialize a list to store the matching combo edges for the current target edge
            matchEdges = []

            for comboEdge in sideEdges[1]:
                # comboP    = comboEdge[:2]  # Start and end points (iX, jX)
                comboP    = [mesh.points[s] for s in comboEdge[:2]]
                comboDist = comboEdge[2]   # Distance between points

                # Check if the points match and the distance is the same, taking into account the direction
                if (np.allclose(np.stack(targetP), np.stack(comboP)      , rtol=tol, atol=tol)  or  # noqa: E272
                    np.allclose(np.stack(targetP), np.stack(comboP[::-1]), rtol=tol, atol=tol)) and \
                    np.isclose(targetDist, comboDist):
                    matchEdges.append(comboEdge)

            # This should result in exactly 1 match
            if len(matchEdges) == 1:
                matches.append((edge, matchEdges.pop()))

        # We only allow 2-1 matches, so in the end we should have exactly 1 match
        if len(matches) != 1:
            return False

        # Here, we only allow 2-1 matches
        # INFO: Uncached version
        # comboEdges  = (e for s in comboSides for e in build_edges(s.corners, mesh.points[s.corners]))
        # INFO: Cached version
        comboEdges = (e for s in comboSides
                        for e in build_edges(arrayToTuple(s.corners), tuple(map(tuple, mesh.points[s.corners]))))
        comboEdges  = find_edge_combinations(comboEdges)

        # Attempt to match the target edges with the candidate edges
        matches     = []  # List to store matching edges

        # Iterate over each target edge
        for targetEdge in targetEdges:
            # targetP    = targetEdge[:2]  # Start and end points (iX, jX)
            targetP    = [mesh.points[s].copy() for s in targetEdge[:2]]
            targetDist = targetEdge[2]   # Distance between points

            if vv is not None:
                # Shift the points in periodic direction
                targetP += vv

            # Initialize a list to store the matching combo edges for the current target edge
            matchEdges = []

            # Iterate over comboEdges to find matching edges
            for comboEdge in comboEdges:
                # comboP    = comboEdge[:2]  # Start and end points (iX, jX)
                comboP    = [mesh.points[s] for s in comboEdge[:2]]
                comboDist = comboEdge[2]   # Distance between points

                # Check if the points match and the distance is the same, taking into account the direction
                if (np.allclose(np.stack(targetP), np.stack(comboP)      , rtol=tol, atol=tol)  or  # noqa: E272
                    np.allclose(np.stack(targetP), np.stack(comboP[::-1]), rtol=tol, atol=tol)) and \
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
                    if np.allclose(mesh.points[p], mesh.points[p1], rtol=tol, atol=tol):
                        matchedPoints += 1

            if matchedPoints == 3:
                matchFound = True
                break

        if not matchFound:
            return False

        # INFO: Uncached version
        # comboEdges  = (e for s in comboSides for e in build_edges(s.corners, mesh.points[s.corners]))
        # INFO: Cached version
        comboEdges = (e for s in comboSides
                        for e in build_edges(arrayToTuple(s.corners), tuple(map(tuple, mesh.points[s.corners]))))
        comboEdges  = find_edge_combinations(comboEdges)

        # Attempt to match the target edges with the candidate edges
        matches     = []  # List to store matching edges

        # Iterate over each target edge
        for targetEdge in targetEdges:
            # targetP    = targetEdge[:2]  # Start and end points (iX, jX)
            targetP    = [mesh.points[s].copy() for s in targetEdge[:2]]
            targetDist = targetEdge[2]   # Distance between points

            if vv is not None:
                # Shift the points in periodic direction
                targetP += vv

            # Initialize a list to store the matching combo edges for the current target edge
            matchEdges = []

            # Iterate over comboEdges to find matching edges
            for comboEdge in comboEdges:
                # comboP    = comboEdge[:2]  # Start and end points (iX, jX)
                comboP    = [mesh.points[s] for s in comboEdge[:2]]
                comboDist = comboEdge[2]  # Distance between points

                # Check if the points match and the distance is the same, taking into account the direction
                if (np.allclose(np.stack(targetP), np.stack(comboP)      , rtol=tol, atol=tol)  or  # noqa: E272
                    np.allclose(np.stack(targetP), np.stack(comboP[::-1]), rtol=tol, atol=tol)) and \
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


# INFO: Uncached version
# def build_edges(corners: np.ndarray, points: np.ndarray) -> list[tuple]:
#     """Build edges from the 4 corners of a quadrilateral, considering CGNS ordering
#     """
#     edges = [
#         (corners[0], corners[1], np.linalg.norm(points[0] - points[1])),  # Edge between points 0 and 1
#         (corners[1], corners[2], np.linalg.norm(points[1] - points[2])),  # Edge between points 1 and 2
#         (corners[2], corners[3], np.linalg.norm(points[2] - points[3])),  # Edge between points 2 and 3
#         (corners[3], corners[0], np.linalg.norm(points[3] - points[0])),  # Edge between points 3 and 0
#     ]
#     return edges


# INFO: Cached version
def arrayToTuple(array: np.ndarray) -> tuple:
    return tuple(array.tolist())


# @cache
@lru_cache(maxsize=65536)
def build_edges(corners: tuple, points : tuple) -> list:
    """Build edges from the 4 corners of a quadrilateral, considering CGNS ordering"""
    edges = [
        (corners[0], corners[1], np.linalg.norm(np.array(points[0]) - np.array(points[1]))),  # Edge between points 0 and 1
        (corners[1], corners[2], np.linalg.norm(np.array(points[1]) - np.array(points[2]))),  # Edge between points 1 and 2
        (corners[2], corners[3], np.linalg.norm(np.array(points[2]) - np.array(points[3]))),  # Edge between points 2 and 3
        (corners[3], corners[0], np.linalg.norm(np.array(points[3]) - np.array(points[0]))),  # Edge between points 3 and 0
    ]
    return edges


# @cache
@lru_cache(maxsize=65536)
def find_edge_combinations(comboEdges) -> list:
    """Build combinations of edges that share exactly one point and form a line
    """
    # Local imports ----------------------------------------
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
                if np.allclose(bbox_min, np.minimum(bbox_min, c1)) and \
                   np.allclose(bbox_max, np.maximum(bbox_max, c1)):
                    # Calculate the distance between the start and end points
                    lineDist = np.linalg.norm(p1 - p2)

                    # Append the indices and the line distance
                    validCombo.append((point1, point2, lineDist))

    return validCombo
