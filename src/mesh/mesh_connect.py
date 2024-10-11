#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of UVWXYZ
#
# Copyright (c) 2022-2024 Andrea Beck
#
# UVWXYZ is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# UVWXYZ is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# UVWXYZ. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import copy
import sys
import traceback
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def flip(side: list, nbside: list) -> int:
    """ Determines the flip of the side-to-side connection
        flip = 1 : 1st node of neighbor side = 1st node of side
        flip = 2 : 2nd node of neighbor side = 1st node of side
        flip = 3 : 3rd node of neighbor side = 1st node of side
        flip = 4 : 4th node of neighbor side = 1st node of side
    """
    # Local imports ----------------------------------------
    from src.common.common import find_index
    # ------------------------------------------------------
    return find_index(nbside, side[0])


def ConnectMesh() -> None:
    # Local imports ----------------------------------------
    import src.io.io_vars as io_vars
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    from src.common.common import find_index
    from src.io.io_vars import MeshFormat
    # ------------------------------------------------------

    match io_vars.outputformat:
        case MeshFormat.FORMAT_HDF5:
            pass
        case _:
            return

    hopout.separator()
    hopout.info('CONNECT MESH...')

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
    # for iSide, side in enumerate(sides):
    #     corners = np.sort(side['Corners'])
    #     corners = hash(corners.tobytes())
    #     side_corners.update({iSide: corners})
    for iElem, elem in enumerate(elems):
        for iSide, side in enumerate(elem['Sides']):
            corners = np.sort(sides[side]['Corners'])
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
    for index, (key, val) in enumerate(corner_side.items()):
        match len(val):
            case 1:  # BC side
                continue
            case 2:  # Internal side
                sideIDs   = val
                corners   = sides[sideIDs[0]]['Corners']
                nbcorners = sides[sideIDs[1]]['Corners']
                flipID    = flip(corners, nbcorners) + 1
                # Connect the sides
                # > Master side contains positive global side ID
                sides[sideIDs[0]].update({'MS'          : 1})
                sides[sideIDs[0]].update({'Connection'  : sideIDs[1]})
                sides[sideIDs[0]].update({'Flip'        : 0})
                sides[sideIDs[0]].update({'nbLocSide'   : sides[sideIDs[1]]['LocSide']})
                # Slave side contains negative global side ID of master side
                sides[sideIDs[1]].update({'MS'          : 0})
                sides[sideIDs[1]].update({'Connection'  : sideIDs[0]})
                sides[sideIDs[1]].update({'Flip'        : flipID})
                sides[sideIDs[1]].update({'nbLocSide'   : sides[sideIDs[0]]['LocSide']})
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
        if key:
            # Get the BCIndex from the list
            bcID = None
            for iBC, bc in enumerate(bcs):
                if key in bc['Name']:
                    bcID = iBC
            if bcID is None:
                # Try again without the leading 'BC_'
                for iBC, bc in enumerate(bcs):
                    if key[:3] == 'BC_' and key[3:] in bc['Name']:
                        bcID = iBC
            if bcID is None:
                hopout.warning('Could not find BC {} in list, exiting...'.format(key))
                traceback.print_stack(file=sys.stdout)
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
                    sides[sideID].update({'BCID': bcID})

    # Try to connect the periodic sides
    # TODO: SET ANOTHER TOLERANCE
    # tol = 5.#5.E-1

    for key, cset in mesh.cell_sets.items():
        # Check if the set is a BC
        if key:
            # Get the BCIndex from the list
            bcID = None
            for iBC, bc in enumerate(bcs):
                if key in bc['Name']:
                    bcID = iBC
            if bcID is None:
                # Try again without the leading 'BC_'
                for iBC, bc in enumerate(bcs):
                    if key[:3] == 'BC_' and key[3:] in bc['Name']:
                        bcID = iBC
            if bcID is None:
                hopout.warning('Could not find BC {} in list, exiting...'.format(key))
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

            # Only periodic BCs
            if bcs[bcID]['Type'][0] != 1:
                continue
            # Only try to connect in positive direction
            if bcs[bcID]['Type'][3] < 0:
                continue
            else:
                iVV = bcs[bcID]['Type'][3] - 1

            # Get the opposite side
            nbType     = copy.copy(bcs[bcID]['Type'])
            nbType[3] *= -1
            nbBCID     = find_index([s['Type'] for s in bcs], nbType)
            nbBCName   = bcs[nbBCID]['Name']

            # Collapse all opposing corner nodes into an [:, 12] array
            nbCellSet  = mesh.cell_sets[nbBCName]

            # Find the mapping to the (N-1)-dim elements
            nbcsetMap = [s for s in range(len(nbCellSet)) if nbCellSet[s] is not None]
            # FIXME: TODO HYBRID MESHES
            if len(nbcsetMap) > 1:
                print('Hybrid meshes currently not supported')
                sys.exit(1)

            # Get the list of sides
            nbFaceSet  = []
            nbmapFaces = []
            nbCorners  = []
            nbPoints   = []
            # for iMap in csetMap:
            nbFaceSet  =  np.array(nbCellSet[csetMap[0]]).astype(int)
            nbmapFaces = mesh.cells[csetMap[0]].data
            nbCorners  = [np.array(nbmapFaces[s - offsetcs]) for s in nbFaceSet]
            nbPoints   = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
            nbPoints   = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
            del nbCorners

            # Build a k-dimensional tree of all points on the opposing side
            stree = spatial.KDTree(nbPoints)

            # Get the list of sides on our side
            iBCsides = np.array(cset[csetMap[0]]).astype(int) - offsetcs

            # Map the unique quad sides to our non-unique elem sides
            for iSide in iBCsides:
                # Get the quad corner nodes
                corners = np.array(nbmapFaces[iSide])
                points  = copy.copy(mesh.points[corners])

                # Shift the points in periodic direction
                for iPoint in range(points.shape[0]):
                    points[iPoint, :] += vvs[iVV]['Dir']
                points = np.sort(points, axis=0)
                points = points.flatten()

                # Query the try for the opposing side
                trSide = copy.copy(stree.query(points))
                del points

                # trSide contains the Euclidean distance and the index of the
                # opposing side in the nbFaceSet
                # tol = np.max(vvs[iVV]['Dir']) * 1.E-1
                tol = np.linalg.norm(vvs[iVV]['Dir'], ord=2) * 1.E-1
                if trSide[0] > tol:
                    hopout.warning('Could not find a periodic side within tolerance {}, exiting...'.format(tol))
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)

                nbiSide  = nbFaceSet[trSide[1]] - offsetcs

                # Get our corner quad nodes
                corners   = np.sort(np.array(nbmapFaces[iSide][0:4]))
                corners   = hash(corners.tobytes())
                sideID    = corner_side[corners][0]
                del corners

                # Get nb corner quad nodes
                nbcorners = np.sort(np.array(nbmapFaces[nbiSide][0:4]))
                nbcorners = hash(nbcorners.tobytes())
                nbSideID  = corner_side[nbcorners][0]
                del nbcorners

                # Build the connection, including flip
                sideIDs   = [sideID, nbSideID]
                points    = mesh.points[sides[sideIDs[0]]['Corners']]
                for iPoint in range(points.shape[0]):
                    points[iPoint, :] += vvs[iVV]['Dir']
                # > Find the first neighbor point to determine the flip
                nbcorners = mesh.points[sides[sideIDs[1]]['Corners']]
                ptree     = spatial.KDTree(nbcorners)

                trCorn    = ptree.query(points[0])
                if trCorn[0] > tol:
                    hopout.warning('Could not determine flip of a periodic side within tolerance {}, exiting...'.format(tol))
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)

                flipID    = trCorn[1] + 1

                # Connect the sides
                # Master side contains positive global side ID
                sides[sideIDs[0]].update({'MS'          : 1})
                sides[sideIDs[0]].update({'Connection'  : sideIDs[1]})
                sides[sideIDs[0]].update({'Flip'        : 0})
                sides[sideIDs[0]].update({'nbLocSide'   : sides[sideIDs[1]]['LocSide']})
                # Slave side contains negative global side ID of master side
                # sides[sideIDs[1]].update({'GlobalSideID': -(sides[sideIDs[0]]['GlobalSideID'])})
                sides[sideIDs[1]].update({'MS'          : 0})
                sides[sideIDs[1]].update({'Connection'  : sideIDs[0]})
                sides[sideIDs[1]].update({'Flip'        : flipID})
                sides[sideIDs[1]].update({'nbLocSide'   : sides[sideIDs[0]]['LocSide']})

    # Non-connected sides without BCID are possible inner sides
    nConnSide = [s for s in sides if 'Connection' not in s and 'BCID' not in s]
    # Append the inner BCs
    for s in (s for s in sides if 'BCID' in s and 'Connection' not in s):
        if mesh_vars.bcs[s['BCID']]['Type'][0] == 0:
            nConnSide.append(s)

    nInterZoneConnect = len(nConnSide)

    hopout.separator()

    # Loop over all sides and try to connect
    while len(nConnSide) > 1:
        # Remove the first side from the list
        targetSide = nConnSide.pop(0)

        # Collapse all opposing corner nodes into an [:, 12] array
        nbCorners  = [s['Corners'] for s in nConnSide]
        nbPoints   = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
        nbPoints   = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
        del nbCorners

        # Build a k-dimensional tree of all points on the opposing side
        stree = spatial.KDTree(nbPoints)

        # Map the unique quad sides to our non-unique elem sides
        corners = targetSide['Corners']
        points  = copy.copy(mesh.points[corners])
        points = np.sort(points, axis=0)
        points = points.flatten()

        # Query the try for the opposing side
        trSide = copy.copy(stree.query(points))
        del points

        # trSide contains the Euclidean distance and the index of the
        # opposing side in the nbFaceSet
        tol = 1.E-10
        if trSide[0] > tol:
            hopout.warning('Could not find an internal side within tolerance {}, exiting...'.format(tol))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

        nbiSide   = trSide[1]

        # Get our corner quad nodes
        corners   = np.sort(targetSide['Corners'])
        corners   = hash(corners.tobytes())
        sideID    = corner_side[corners][0]
        del corners

        # Get nb corner quad nodes
        nbcorners = np.sort(nConnSide[nbiSide]['Corners'])
        nbcorners = hash(nbcorners.tobytes())
        nbSideID  = corner_side[nbcorners][0]
        del nbcorners

        # Build the connection, including flip
        sideIDs   = [sideID, nbSideID]
        points    = mesh.points[sides[sideIDs[0]]['Corners']]
        # > Find the first neighbor point to determine the flip
        nbcorners = mesh.points[sides[sideIDs[1]]['Corners']]
        ptree     = spatial.KDTree(nbcorners)

        trCorn    = ptree.query(points[0])
        if trCorn[0] > tol:
            hopout.warning('Could not determine flip of an internal side within tolerance {}, exiting...'.format(tol))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        flipID = trCorn[1] + 1

        # Connect the sides
        # Master side contains positive global side ID
        sides[sideIDs[0]].update({'MS'          : 1})
        sides[sideIDs[0]].update({'Connection'  : sideIDs[1]})
        sides[sideIDs[0]].update({'Flip'        : 0})
        sides[sideIDs[0]].update({'nbLocSide'   : sides[sideIDs[1]]['LocSide']})
        # Slave side contains negative global side ID of master side
        sides[sideIDs[1]].update({'MS'          : 0})
        sides[sideIDs[1]].update({'Connection'  : sideIDs[0]})
        sides[sideIDs[1]].update({'Flip'        : flipID})
        sides[sideIDs[1]].update({'nbLocSide'   : sides[sideIDs[0]]['LocSide']})

        # Update the list
        nConnSide = [s for s in sides if 'Connection' not in s and 'BCID' not in s]
        # Append the inner BCs
        for s in (s for s in sides if 'BCID' in s and 'Connection' not in s):
            if mesh_vars.bcs[s['BCID']]['Type'][0] == 0:
                nConnSide.append(s)

    if nInterZoneConnect > 0:
        hopout.sep()
        hopout.routine('Connected {} inter-zone faces'.format(nInterZoneConnect))

    # Set the global side ID
    globalSideID = 0
    for iSide, side in enumerate(sides):
        # Already counted the side
        if 'GlobalSideID' in side:
            continue

        if 'Connection' not in side:  # BC side
            side.update({'GlobalSideID':  globalSideID+1 })
            globalSideID += 1
        else:                         # Internal / periodic side
            # Master side does not have a flip
            if side['MS'] == 1:
                # Set the positive globalSideID of the master side
                side.update({'GlobalSideID':  globalSideID+1 })
                # Set the negative globalSideID of the slave  side
                nbSideID = side['Connection']
                sides[nbSideID].update({'GlobalSideID': -(globalSideID+1)})
            globalSideID += 1

    # Count the sides
    nsides         = len(sides)
    ninnersides    = 0
    nbcsides       = 0
    nperiodicsides = 0
    for iSide, side in enumerate(sides):
        if 'Connection' in side:
            if 'BCID' in side:
                nperiodicsides += 1
            else:
                ninnersides    += 1
        else:
            if 'BCID' in side:
                nbcsides       += 1
            else:
                hopout.warning('Found unconnected side which is not a BC side, exiting...')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

    hopout.sep()
    hopout.info(' Number of sides          : {:12d}'.format(nsides))
    hopout.info(' Number of inner sides    : {:12d}'.format(ninnersides))
    hopout.info(' Number of boundary sides : {:12d}'.format(nbcsides))
    hopout.info(' Number of periodic sides : {:12d}'.format(nperiodicsides))
    hopout.sep()

    hopout.info('CONNECT MESH DONE!')
