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
import bisect
import copy
import itertools
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from typing import Optional, Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# import meshio
import numpy as np
from numpy.linalg import norm
from scipy.spatial import KDTree
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.output.output as hopout
import pyhope.mesh.mesh_vars as mesh_vars
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
mortarToCorners = { 1: (0, 1, 3, 2),  # 4-1 mortar
                    2: (0, 3),        # 2-1 mortar, split in eta
                    3: (0, 2)         # 2-1 mortar, split in xi
                  }
# ==================================================================================================================================


def ConnectMortar( nConnSide  : list
                 , nConnCenter: list
                 # , mesh       : meshio.Mesh
                 , elems      : list
                 , sides      : list
                 , bar) -> tuple[list, list]:
    """ Function to connect mortar sides

        Args:
            doPeriodic: Flag to enable periodic connections
    """
    # Local imports ----------------------------------------
    from pyhope.mesh.connect.connect_rbtree import LinkOffsetManager, RedBlackTree
    from pyhope.common.common_tools import IndexedLists
    # ------------------------------------------------------

    if len(nConnSide) == 0:
        return elems, sides

    # Change the title of the progress bar
    bar.title('│               Preparing Mortars')

    # Cache mesh points for performance
    points: Final[np.ndarray] = mesh_vars.mesh.points

    # Set BC and periodic sides
    bcs: Final[list[type | None]] = mesh_vars.bcs
    vvs: Final[list             ] = mesh_vars.vvs

    # Build a k-dimensional tree of all points on the opposing side
    ctree:     Final[KDTree      ] = KDTree(np.array(nConnCenter))
    indexList: Final[IndexedLists] = IndexedLists()

    for nConnID, (side, center) in enumerate(zip(nConnSide, nConnCenter)):
        targetSide   = side
        targetCenter = copy.copy(center)

        # Get the opposite side
        bcID = targetSide.bcid
        if bcID is not None and bcs[bcID].type[0] == 1:
            iVV    = bcs[bcID].type[3]
            VV     = vvs[np.abs(iVV)-1]['Dir'] * np.sign(iVV)
            # Shift the center in periodic direction
            targetCenter += VV

        # Calculate the radius of the convex hull
        targetRadius    = norm(np.ptp(points[targetSide.corners], axis=0)) / 2.

        # Get all potential mortar neighbors within the radius
        targetNeighbors = tuple(s for s in ctree.query_ball_point(targetCenter, targetRadius) if nConnSide[s].elemID != targetSide.elemID)  # noqa: E501
        indexList .add(nConnID, targetNeighbors)

    # Obtain the target side IDs
    targetSides:   Final[list[int]] = [s for s in indexList.data.keys() if len(indexList.data[s]) > 0]
    # Create a global offset manager.
    offsetManager: Final[LinkOffsetManager] = LinkOffsetManager()
    # Convert the sides to a red-black tree
    rbtsides:      Final[RedBlackTree     ] = RedBlackTree.from_list(sides, offsetManager)

    # Change the title of the progress bar
    bar.title('│              Processing Mortars')

    for targetID in targetSides:
        # Skip already connected sides
        # if indexList.data[targetID] == -1:
        if targetID not in indexList.data.keys():
            continue

        # Get the target neighbors
        targetNeighbors = indexList.data[targetID]

        # Skip elements with zero or one neighbors
        if len(targetNeighbors) < 2:
            continue

        targetSide   = nConnSide[  targetID]
        targetCenter = nConnCenter[targetID]

        # Get the opposite side
        bcID = targetSide.bcid if targetSide.bcid is not None and bcs[targetSide.bcid].type[0] == 1 else None

        # Prepare combinations for 2-to-1 and 4-to-1 mortar matching
        candidate_combinations = list(itertools.combinations(targetNeighbors, 2))
        if len(targetNeighbors) >= 4:
            candidate_combinations += list(itertools.combinations(targetNeighbors, 4))

        # Attempt to match the target side with candidate combinations
        comboSides   = ()
        for comboIDs in candidate_combinations:
            # Get the candidate sides
            comboSides   = tuple(nConnSide[iSide] for iSide in comboIDs)

            # Check if we found a valid match
            if not find_mortar_match(targetSide.corners, comboSides, bcID):
                continue

            # Get our and neighbor corner quad nodes
            sideID   = targetSide.sideID
            nbSideID = tuple(side.sideID for side in comboSides)

            # Build the connection, including flip
            sideIDs  = (sideID, nbSideID)

            # Connect mortar sides and update the list
            # connect_mortar_sides(sideIDs, elems, sides, rbtsides, offsetManager, bcID)
            connect_mortar_sides(sideIDs, elems, rbtsides, offsetManager, bcID)

            # Remove the target side from the list
            removeSides = [targetID] + list(comboIDs)
            # for r in removeSides:
            #     indexList.data[r] = -1
            indexList.remove_index(removeSides)

            # Update the progress bar
            bar.step(len(nbSideID) + 1)

            # Break out of the loop
            break

    # Change the title of the progress bar
    bar.title('│              Finalizing Mortars')

    # Convert sides back to a list
    sides = rbtsides.to_list()

    # Perform explicit clean-up
    del rbtsides
    del offsetManager

    # Also update the elems with the new side IDs
    # > First, build a dictionary mapping elemID to list of sideIDs
    elem_to_side_ids = defaultdict(list)
    for side in sides:
        elem_to_side_ids[side.elemID].append(side.sideID)

    # > Then update elems using the dictionary
    for elem in elems:
        elem.sides = np.array(elem_to_side_ids.get(elem.elemID, []))

    # Change the title of the progress bar
    bar.title('│                Processing Sides')

    return elems, sides


def connect_mortar_sides( sideIDs    : tuple
                        , elems      : list
                        , rbtsides
                        , offsetManager
                        , bcID         : Optional[int] = None) -> None:
    """ Connect the master (big mortar) and the slave (small mortar) sides
        > Create the virtual sides as needed
    """
    # Local imports ----------------------------------------
    from pyhope.mesh.connect.connect_rbtree import SideNode
    from pyhope.mesh.connect.connect import flip_analytic
    from pyhope.mesh.mesh_common import type_to_mortar_flip
    # ------------------------------------------------------

    # Get the master and slave sides
    masterSide    = rbtsides[sideIDs[0] + offsetManager.get_offset(sideIDs[0])].value
    masterElem    = elems[masterSide.elemID]
    masterCorners = masterSide.corners

    if bcID is not None:
        bcName        = mesh_vars.bcs[bcID].name
        masterCorners = np.fromiter((mesh_vars.periNodes[(s, bcName)] for s in masterCorners), dtype=int)

    # Convert to hashable tuple
    masterCorners = tuple(masterCorners)

    # Build mortar type and orientation
    nMortars   = len(sideIDs[1])
    slaveSides = tuple(rbtsides[s + offsetManager.get_offset(s)].value for s in sideIDs[1])

    match nMortars:
        case 2:
            # Check which edges of big and small side are identical to determine the mortar type
            slaveSide    = slaveSides[0]
            slaveCorners = tuple(slaveSide.corners)

            # Check which edges match
            # INFO: Uncached version
            if   points_exist_in_target((masterCorners[0], masterCorners[1]), slaveCorners) or \
                 points_exist_in_target((masterCorners[2], masterCorners[3]), slaveCorners):  # noqa: E271
                mortarType = 2
            elif points_exist_in_target((masterCorners[1], masterCorners[2]), slaveCorners) or \
                 points_exist_in_target((masterCorners[0], masterCorners[3]), slaveCorners):
                mortarType = 3
            else:
                hopout.error('Could not determine mortar type, exiting...', traceback=True)

            del slaveSide
            del slaveCorners

            # Sort the small sides
            slaveSides = tuple(s for i in [0, 2]
                                 for s in slaveSides if points_exist_in_target((masterCorners[i],), tuple(s.corners)))

        case 4:
            mortarType = 1
            # Sort the small sides
            slaveSides = tuple(s for i in [0, 1, 3, 2]
                                 for s in slaveSides if points_exist_in_target((masterCorners[i],), tuple(s.corners)))

        case _:
            hopout.error('Found invalid number of sides for mortar side, exiting...', traceback=True)

    # Sanity check
    if len(slaveSides) != nMortars:
        hopout.error('Could not determine mortar type, exiting...', traceback=True)

    # Update the master side
    masterSide.MS          = 1            # noqa: E251
    masterSide.connection  = -mortarType  # noqa: E251
    masterSide.flip        = 0            # noqa: E251
    masterSide.nbLocSide   = 0            # noqa: E251

    flipMap = type_to_mortar_flip(mesh_vars.elems[masterSide.elemID].type)

    # Map mortar types to their corresponding corner lists.
    mortarCorners = mortarToCorners[mortarType]

    # Precompute mappings and indices.
    masterElemID  = masterElem.elemID
    masterSideID  = masterSide.sideID + offsetManager.get_offset(masterSide.sideID)

    # Build lists of new SIDE objects and their sideIDs.
    new_sides    = []
    new_sideIDs  = []

    SIDE = mesh_vars.SIDE

    for i, slave in enumerate(slaveSides):
        corner   = mortarCorners[i]
        sCorners = slave.corners
        sideType = 100 + len(sCorners)
        flipID   = flip_analytic(masterCorners[corner], sCorners) + 1
        flipID   = flipMap.get(corner, {}).get(flipID, flipID)
        slave.flip = flipID  # update slave side's flip

        newID = masterSideID + i + 1
        side  = SIDE(
                  sideType   = sideType,            # noqa: E251
                  elemID     = masterElemID,        # noqa: E251
                  sideID     = newID,               # noqa: E251
                  locSide    = masterSide.locSide,  # noqa: E251
                  locMortar  = i + 1,               # noqa: E251
                  MS         = 1,                   # noqa: E251
                  flip       = flipID,              # noqa: E251
                  connection = slave.sideID,        # noqa: E251
                  nbLocSide  = slave.locSide        # noqa: E251
                )
        new_sides  .append(side)
        new_sideIDs.append(newID)

        # Update the slave side: connect to the master
        slave.connection = masterSideID
        slave.sideType   = -sideType
        slave.MS         = 0
        slave.flip       = flipID
        # Update the link in the red-black tree
        rbtsides[slave.sideID + offsetManager.get_offset(slave.sideID)].link = masterSide.sideID

    # Insert the new sides into the red-black tree
    # Each insertion automatically notifies offsetManager to shift subsequent indices.
    for i, new_side in enumerate(new_sides):
        new_node = SideNode(value=new_side, link=new_side.connection)
        insertion_index = masterSideID + 1 + i
        rbtsides.insert(insertion_index, new_node)

        # Insert the new sideID into the element's side list
        bisect.insort(elems[masterElemID].sides, insertion_index)


def find_mortar_match( targetCorners: np.ndarray
                     , comboSides   : tuple
                     # , mesh         : meshio.Mesh
                     , bcID         : Optional[int] = None) -> bool:
    """ Check if the combined points of candidate sides match the target side within tolerance.
    """

    points: Final[np.ndarray] = mesh_vars.mesh.points

    # Passing a bcID means we are dealing with periodic boundaries
    if bcID is not None:
        bcName        = mesh_vars.bcs[bcID].name
        targetCorners = np.fromiter((mesh_vars.periNodes[(s, bcName)] for s in targetCorners), dtype=int)

    # Check if exactly one combo point matches each target point
    unmatchedCorners = set(targetCorners)
    for side in comboSides:
        for c in side.corners:
            if c in unmatchedCorners:
                unmatchedCorners.remove(c)
                # Found all target corners in this inner loop.
                if not unmatchedCorners:
                    break
        # Found all target corners overall
        if not unmatchedCorners:
            break

    if unmatchedCorners:
        return False

    # PERF: Alternative implementation, about the same speed
    # targetSet = set(targetCorners)
    # comboSet  = set()
    #
    # for side in comboSides:
    #     # Update with all corners from this side.
    #     comboSet.update(side.corners)
    #     # If we've already seen all target corners, break early
    #     if targetSet.issubset(comboSet):
    #         break
    #
    # # If any target corner is missing, return False
    # if not targetSet.issubset(comboSet):
    #     return False

    # Build the target edges
    # INFO: Uncached version
    targetEdges = build_edges(targetCorners, points[targetCorners])
    # INFO: Cached version
    # targetEdges = build_edges(arrayToTuple(targetCorners), tuple(map(tuple, points[targetCorners])))
    matches     = []

    # First, check for 2-1 matches
    if len(comboSides) == 2:
        # Look for 2-1 matches, we need exactly one common edge
        # INFO: Uncached version
        comboEdges  = (e for s in comboSides for e in build_edges(s.corners, points[s.corners]))
        # INFO: Cached version
        # comboEdges = (e for s in comboSides
        #                 for e in build_edges(arrayToTuple(s.corners), tuple(map(tuple, points[s.corners]))))
        comboEdges = find_edge_combinations(comboEdges)

        # Attempt to match the target edges with the candidate edges
        matches     = []  # List to store matching edges

        # Iterate over each target edge
        # > Start and end points (iX, jX), distance between points
        for *targetEdge, targetDist in targetEdges:
            # Convert the star-unpacked targetEdge [list] into a tuple
            targetEdge = tuple(targetEdge)
            # Find the matching combo edges for the current target edge
            matchEdges = [e for e in comboEdges if (targetEdge[:2] == e[:2] or targetEdge[:2] == e[1::-1]) and
                                                   np.isclose(targetDist, e[2])]

            # We only allow 2-1 matches, so in the end we should have exactly 1 match
            if len(matchEdges) > 1:
                return False
            elif len(matchEdges) == 1:
                matches.append((targetEdge, matchEdges.pop()))

        if len(matches) != 2:
            return False

    # Next, check for 4-1 matches
    if len(comboSides) == 4:
        # Check if there is exactly one point that all 4 sides have in common.
        # common_points = set(comboSides[0].corners)
        # matchFound    = any(sum(p in side.corners for side in comboSides[1:]) == 3 for p in common_points)
        #
        # if not matchFound:
        #     return False

        common = set(comboSides[0].corners).intersection(*(side.corners for side in comboSides[1:]))
        # Enforce exactly one common point
        if len(common) != 1:
            return False

        # INFO: Uncached version
        comboEdges  = (e for s in comboSides for e in build_edges(s.corners, points[s.corners]))
        # INFO: Cached version
        # comboEdges = (e for s in comboSides
        #                 for e in build_edges(arrayToTuple(s.corners), tuple(map(tuple, points[s.corners]))))
        comboEdges = find_edge_combinations(comboEdges)

        # Attempt to match the target edges with the candidate edges
        matches     = []  # List to store matching edges

        # Iterate over each target edge
        # > Start and end points (iX, jX), distance between points
        for *targetEdge, targetDist in targetEdges:
            # Convert the star-unpacked targetEdge [list] into a tuple
            targetEdge = tuple(targetEdge)
            # Find the matching combo edges for the current target edge
            matchEdges = [e for e in comboEdges if (targetEdge[:2] == e[:2] or targetEdge[:2] == e[1::-1]) and
                                                   np.isclose(targetDist, e[2])]

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
# def points_exist_in_target(pts: tuple, slavePts: tuple) -> bool:
#     """ Check if the combined points of candidate sides match the target side
#     """
#     # return np.all(np.isin(pts, slavePts))
#     return set(pts).issubset(set(slavePts))


# INFO: Cached version
@lru_cache(maxsize=65536)
def points_exist_in_target(pts: tuple, slavePts: tuple) -> bool:
    """ Check if the combined points of candidate sides match the target side
    """
    # return np.all(np.isin(pts, slavePts))
    return set(pts).issubset(set(slavePts))


# INFO: Uncached version
def build_edges(corners: np.ndarray, points: np.ndarray) -> tuple:
    """Build edges from the 4 corners of a quadrilateral, considering CGNS ordering
    """
    edges = ((corners[0], corners[1], norm(np.array(points[0]) - np.array(points[1]))),  # Edge between points 0 and 1
             (corners[1], corners[2], norm(np.array(points[1]) - np.array(points[2]))),  # Edge between points 1 and 2
             (corners[2], corners[3], norm(np.array(points[2]) - np.array(points[3]))),  # Edge between points 2 and 3
             (corners[3], corners[0], norm(np.array(points[3]) - np.array(points[0]))))  # Edge between points 3 and 0
    return edges


# INFO: Cached version
# def arrayToTuple(array: np.ndarray) -> tuple:
#     return tuple(array.tolist())
#
#
# # @cache
# @lru_cache(maxsize=65536)
# def build_edges(corners: tuple, points: np.ndarray) -> list[tuple]:
#     """Build edges from the 4 corners of a quadrilateral, considering CGNS ordering
#     """
#     edges = [
#         (corners[0], corners[1], norm(np.array(points[0]) - np.array(points[1]))),  # Edge between points 0 and 1
#         (corners[1], corners[2], norm(np.array(points[1]) - np.array(points[2]))),  # Edge between points 1 and 2
#         (corners[2], corners[3], norm(np.array(points[2]) - np.array(points[3]))),  # Edge between points 2 and 3
#         (corners[3], corners[0], norm(np.array(points[3]) - np.array(points[0]))),  # Edge between points 3 and 0
#     ]
#     return edges


# @cache
@lru_cache(maxsize=65536)
def find_edge_combinations(comboEdges) -> tuple:
    """Build combinations of edges that share exactly one point and form a line
    """
    points = mesh_vars.mesh.points

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
                edgePoints = np.array((i1, j1, i2, j2))

                # Find the index of the common point and delete it
                # commonIndex = np.where(edgePoints == commonPoint)[0]
                # edgePoints  = np.delete(edgePoints, commonIndex)
                edgePoints = edgePoints[edgePoints != commonPoint]

                # The remaining points are the start and end points of the edge combination
                point1, point2 = edgePoints

                # Get the coordinates of the points
                p1, p2 = points[point1], points[point2]

                # INFO: This is a more strict check that is not necessary
                # c1 = points[commonPoint]
                #
                # # Calculate the bounding box of the two edge points
                # bbox_min = np.minimum(p1, p2)
                # bbox_max = np.maximum(p1, p2)
                #
                # Check if the common point is within the bounding box of p1 and p2
                # if np.allclose(bbox_min, np.minimum(bbox_min, c1)) and \
                #    np.allclose(bbox_max, np.maximum(bbox_max, c1)):
                #     # Calculate the distance between the start and end points
                #     lineDist = np.linalg.norm(p1 - p2)
                #
                #     # Append the indices and the line distance
                #     validCombo.append((point1, point2, lineDist))

                lineDist = norm(p1 - p2)
                validCombo.append((point1, point2, lineDist))

    return tuple(validCombo)
