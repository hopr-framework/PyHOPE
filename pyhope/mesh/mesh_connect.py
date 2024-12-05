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
from typing import Union, Tuple, cast
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


def flip_analytic(side: list, nbside: list) -> int:
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

    flipID    = trCorn[1] + 1
    return flipID


def connect_sides(sideIDs: list, sides: list, flipID: int) -> None:
    """ Connect the master and slave sides
    """
    sides[sideIDs[0]].update(
        # Master side contains positive global side ID
        MS         = 1,                         # noqa: E251
        connection = sideIDs[1],                # noqa: E251
        flip       = flipID,                    # noqa: E251
        nbLocSide  = sides[sideIDs[1]].locSide  # noqa: E251
    )
    sides[sideIDs[1]].update(
        MS         = 0,                         # noqa: E251
        connection = sideIDs[0],                # noqa: E251
        flip       = flipID,                    # noqa: E251
        nbLocSide  = sides[sideIDs[0]].locSide  # noqa: E251
    )


def connect_mortar_sides(sideIDs: list, elems: list, sides: list) -> None:
    """ Connect the master (big mortar) and the slave (small mortar) sides
        > Create the virtual sides as needed
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_vars import SIDE
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
    sides[sideIDs[0]].update(
        # Master side contains positive global side ID
        MS          = 1,            # noqa: E251
        connection  = -mortarType,  # noqa: E251
        flip        = 0,            # noqa: E251
        nbLocSide   = 0,            # noqa: E251
    )

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

    # Insert the virtual sides
    # for key, val in enumerate(slaveSides):

    for key, val in enumerate(slaveSides):
        tol        = mesh_vars.tolInternal
        points     = mesh_vars.mesh.points[masterSide.corners]
        nbcorners  = mesh_vars.mesh.points[val.corners]
        match mortarType:
            case 1:  # 4-1 mortar
                mortarCorners = [0, 1, 3, 2]  # Prepare for non-quad mortars
                flipID = flip_physical(points[mortarCorners[key]], nbcorners, tol, 'mortar')
                # Correct for the corner offset
                flipID = (flipID - key + 1) % 4
            case 2:  # 2-1 mortar, split in eta
                mortarCorners = [0, -1]  # Prepare for non-quad mortars
                flipID = flip_physical(points[mortarCorners[key]], nbcorners, tol, 'mortar')
                # Correct for the corner offset
                mortarLength  = [0, 1]
                flipID = (flipID - mortarLength[key] + 4) % 4
            case 3:  # 2-1 mortar, split in xi
                mortarCorners = [0, -2]  # Prepare for non-quad mortars
                flipID = flip_physical(points[mortarCorners[key]], nbcorners, tol, 'mortar')
                # Correct for the corner offset
                mortarLength  = [0, 2]
                flipID = (flipID - mortarLength[key] + 4) % 4
        val.update(flip=flipID)

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
        val.update(connection = masterSide.sideID,             # noqa: E251
                   # Small sides are always slave sides
                   sideType   = -104,                          # noqa: E251
                   MS         = 0,                             # noqa: E251
                   flip       = flipID,                        # noqa: E251
                  )


def find_bc_index(bcs: list, key: str) -> Union[int, None]:
    """ Find the index of a BC from its name in the list of BCs
    """
    for iBC, bc in enumerate(bcs):
        if key in bc.name:
            return iBC
        # Try again without the leading 'BC_'
        if key[:3] == 'BC_' and key[3:] in bc.name:
            return iBC
    return None


def find_closest_side(points: np.ndarray, stree: spatial._kdtree.cKDTree, tol: float, msg: str, doMortars: bool = False) -> int:
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
    return trSide[1]


def find_mortar_match(targetPoints: np.ndarray, comboPoints: np.ndarray, tol: float) -> bool:
    """ Check if the combined points of candidate sides match the target side within tolerance.
    """
    ttree = spatial.KDTree(targetPoints)
    distances, indices = ttree.query(comboPoints)

    # At least one combo point must match each target point
    matchedIndices = np.unique(indices[distances <= tol])
    if len(matchedIndices) < 4:
        return False

    # Check if exactly one combo point matches each target point
    for point in targetPoints:
        distancesToPoints = np.linalg.norm(comboPoints - point, axis=1)
        if np.sum(distancesToPoints <= tol) != 1:
            return False

    # Build the remaining points
    unmatchedPoints  = comboPoints[distances > tol]

    # Check if there are at no single combo points, i.e., the small sides are connected
    if len(np.unique(unmatchedPoints, axis=0)) < 2:
        return False

    # For each remaining comboPoint, ensure it matches at least one other remaining comboPoint
    for i, point in enumerate(unmatchedPoints):
        otherPoints       = np.delete(unmatchedPoints, i, axis=0)
        distancesToPoints = np.linalg.norm(otherPoints - point, axis=1)
        # We allow only 2-1 and 4-1 matches, so either 1 or 3 points must match
        match np.sum(distancesToPoints < tol):
            case 1 | 3:
                pass
            case _:
                return False

    # Found a valid match
    return True


def get_side_id(corners: np.ndarray, side_dict: dict) -> int:
    """ Get sorted corners and hash them to get the side ID
    """
    corners_sorted = np.sort(corners)
    corners_hash = hash(corners_sorted.tobytes())
    return side_dict[corners_hash][0]


def get_nonconnected_sides(sides: list, mesh: meshio.Mesh) -> Tuple[list, list]:
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


def ConnectMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.common.common import find_index
    from pyhope.io.io_vars import MeshFormat
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    match io_vars.outputformat:
        case MeshFormat.FORMAT_HDF5:
            pass
        case _:
            return

    hopout.separator()
    hopout.info('CONNECT MESH...')
    hopout.sep()

    mesh_vars.doMortars = GetLogical('doMortars')
    doMortars = mesh_vars.doMortars

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
                sides[sideID].update(bcid=bcID)

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
            nbSideIdx = find_closest_side(points, stree, tol, 'periodic')
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

    # Non-connected sides without BCID are possible inner sides
    nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)
    nInterZoneConnect      = len(nConnSide)

    hopout.separator()

    # Loop over all sides and try to connect
    iter    = 0
    maxIter = copy.copy(len(nConnSide))
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
        tol        = mesh_vars.tolInternal
        nbSideIdx  = find_closest_side(points, stree, tol, 'internal', doMortars)

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

        # Mortar side
        # > Here, we can only attempt to connect big to small mortar sides. Thus, if we encounter a small mortar sides which
        # > generates no match, we simply append the side again at the end and try again. As the loop exists after checking
        # > len(nConnSide), we will check each side once.
        else:
            if not doMortars:
                hopout.warning(f'Could not find internal side within tolerance {tol}, exiting...')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

            # Calculate the radius of the convex hull
            targetPoints = mesh.points[targetSide.corners]
            targetMinMax = (targetPoints.min(axis=0), targetPoints.max(axis=0))
            targetRadius = np.linalg.norm(targetMinMax[1] - targetMinMax[0], ord=2) / 2.

            # Find nearby sides to consider as candidate mortar sides
            targetNeighbors = ctree.query_ball_point(targetCenter, targetRadius)

            # Prepare combinations for 2-to-1 and 4-to-1 mortar matching
            candidate_combinations = []
            if len(targetNeighbors) >= 2:
                candidate_combinations += list(itertools.combinations(targetNeighbors, 2))
            if len(targetNeighbors) >= 4:
                candidate_combinations += list(itertools.combinations(targetNeighbors, 4))

            # Attempt to match the target side with candidate combinations
            matchFound   = False
            comboSides   = []
            comboCorners = []
            for comboIDs in candidate_combinations:
                # Get the candidate sides
                comboSides   = [nConnSide[iSide] for iSide in comboIDs]
                comboCorners = [s.corners for s in comboSides]
                comboPoints  = np.concatenate([mesh.points[c] for c in comboCorners], axis=0)

                # Found a valid match
                # print(find_mortar_match(targetPoints, comboPoints, tol))
                if find_mortar_match(targetPoints, comboPoints, tol):
                    matchFound = True
                    break

            if matchFound:
                # Get our and neighbor corner quad nodes
                # sideID    =  get_side_id(targetSide['Corners'], corner_side)
                # nbSideID  = [get_side_id(cast(np.ndarray, c)  , corner_side) for c in comboCorners]
                sideID    =  targetSide.sideID
                nbSideID  = [side.sideID for side in comboSides]

                # Build the connection, including flip
                sideIDs   = [sideID, nbSideID]

                # Connect the sides
                connect_mortar_sides(sideIDs, elems, sides)

                # Update the list
                nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)

            # No connection, attach the side at the end
            else:
                nConnSide  .append(targetSide)
                nConnCenter.append(targetCenter)

    nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)
    if len(nConnSide) > 0:
        hopout.warning('Could not connect {} side{}'.format(len(nConnSide), '' if len(nConnSide) == 1 else 's'))

        for side in nConnSide:
            hopout.info('SideID : {}'.format(side.sideID))
            hopout.info('{}'.format(side.dict()))
            hopout.info('Corners:')
            print('{}'.format(mesh.points[side.corners]))
        sys.exit(1)

    # TODO: Check if mortars are watertight

    if nInterZoneConnect > 0:
        hopout.sep()
        hopout.routine('Connected {} inter-zone faces'.format(nInterZoneConnect))

    # Set the global side ID
    globalSideID = 0
    for iSide, side in enumerate(sides):
        # Already counted the side
        if side.globalSideID is not None:
            continue

        globalSideID += 1
        side.update(globalSideID=globalSideID)
        if side.connection is None:         # BC side
            pass
        elif side.connection < 0:           # Big mortar side
            pass
        elif side.MS == 1:                  # Internal / periodic side (master side)
            # Master side does not have a flip
            # Set the positive globalSideID of the master side
            # side.update({'GlobalSideID':  globalSideID })
            # Set the negative globalSideID of the slave  side
            nbSideID = side.connection
            sides[nbSideID].update(globalSideID=-(globalSideID))

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
