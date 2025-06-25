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
# import copy
import gc
import sys
from collections import defaultdict
from typing import Final, Optional, cast
# from multiprocessing import Pool
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.output.output as hopout
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def flip_analytic(side: int, nbside: np.ndarray) -> int:
    """ Determines the flip of the side-to-side connection based on the analytic side ID
        flip = 1 : 1st node of neighbor side = 1st node of side
        flip = 2 : 2nd node of neighbor side = 1st node of side
        flip = 3 : 3rd node of neighbor side = 1st node of side
        flip = 4 : 4th node of neighbor side = 1st node of side
    """
    # Local imports ----------------------------------------
    from pyhope.common.common import find_index
    # ------------------------------------------------------
    return find_index(nbside, side)


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


def find_bc_index(bcs: list, key: str) -> Optional[int]:
    """ Find the index of a BC from its name in the list of BCs
    """
    for iBC, bc in enumerate(bcs):
        if key in bc.name:
            return iBC
        # Try again without the leading 'BC_'
        if key[:3] == 'BC_' and key[3:] in bc.name or \
           key[:3] == 'bc_' and key[3:] in bc.name:
            return iBC
    return None


def find_closest_side(points: np.ndarray, stree: KDTree, tol: float, msg: str, doMortars: bool = False) -> int:
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

        hopout.error(f'Could not find {msg} side within tolerance {tol}, exiting...', traceback=True)
    return cast(int, trSide[1])


def get_side_id(corners: np.ndarray, side_dict: dict) -> int:
    """ Get sorted corners and hash them to get the side ID
    """
    corners_hash = hash(np.sort(corners).tobytes())
    return side_dict[corners_hash][0]


def get_nonconnected_sides(sides: list, mesh: meshio.Mesh) -> tuple[list, list[np.ndarray]]:
    """ Get a list of internal sides that are not connected to any
        other side together with a list of their centers
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------
    # Update the list
    nConnSide   = [s for s in sides if   s.connection is None  # noqa: E271
                                    and (s.bcid is None or mesh_vars.bcs[s.bcid].type[0] in (0, 1))]

    nConnCenter = [np.mean(mesh.points[s.corners], axis=0) for s in nConnSide]
    return nConnSide, nConnCenter


def periodic_update(sides: tuple, elems: tuple, vv: np.ndarray) -> None:
    """Update the mesh after connecting periodic sides
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_common import sidetovol2
    # ------------------------------------------------------
    # Periodic corrections are only supported for hexahedral elements
    if elems[0].type % 100 != 8 or elems[1].type % 100 != 8:
        return

    nGeo:   Final[int]         = mesh_vars.nGeo
    points: Final[npt.NDArray] = mesh_vars.mesh.points
    tol:    Final[float]       = mesh_vars.tolPeriodic

    # for iy, ix in np.ndindex(nodes.shape[:2]):
    #     node   = nodes[ix, iy]
    #     nbNode = nbNodes[indices[ix, iy, 0], indices[ix, iy, 1]]
    #
    #     # Sanity check if the periodic vector matches
    #     if not np.allclose(vv['Dir'], mesh_vars.mesh.points[nbNode] - mesh_vars.mesh.points[node],
    #                        rtol=mesh_vars.tolPeriodic, atol=mesh_vars.tolPeriodic):
    #         hopout.error('Error in periodic update, periodic vector does not match!')
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

    # Map the meshio nodes to the tensor-product nodes
    elemType = elems[0].type
    nodes    = elems[0].nodes[sidetovol2(nGeo, 0            , sides[0].face, elemType)]
    nbNodes  = elems[1].nodes[sidetovol2(nGeo, sides[1].flip, sides[1].face, elemType)]

    # INFO: THIS CURRENTLY MIGHT NOT WORK SINCE WE POTENTIALLY ONLY HAVE THE CORNER NODES AVAILABLE
    try:
        # Translate to periodic nodes
        nbCheck = np.vectorize(lambda s: mesh_vars.periNodes[(s, mesh_vars.bcs[sides[1].bcid].name)], otypes=[int])(nbNodes)

        # Check if the node IDs match
        if not np.array_equal(nodes, nbCheck):
            # Print the node IDs
            print(hopout.warn(f'NodeIDs side[-]: {nodes  }'))
            print(hopout.warn(f'NodeIDs side[+]: {nbCheck}'))
            hopout.error('Error in periodic update, node IDs do not match!')
    # Fallback to comparison of physical coordinates
    except KeyError:
        # Check if periodic vector matches using vectorized np.allclose
        if not np.allclose(points[nodes] + vv, points[nbNodes], rtol=tol, atol=tol):
            # Print the node coordinates
            print(hopout.warn(f'Coordinates side[-]: {points[nodes]  }'))
            print(hopout.warn(f'Coordinates side[+]: {points[nbNodes]}'))
            hopout.error('Error in periodic update, periodic vector does not match!')

    # Calculate the center for both points
    centers = 0.5 * (points[nodes] + points[nbNodes])

    # Update the mesh points for both node and nbNode
    points[nodes]   = centers - 0.5*vv
    points[nbNodes] = centers + 0.5*vv


# PERF: The parallel version does not really speed up the process
# def init_side_worker(s) -> None:
#     """ Initialize the worker with the side data
#     """
#     global worker_sides
#     worker_sides = s


# PERF: The parallel version does not really speed up the process
# def compute_side_hash(side) -> tuple:
#     """ Compute the hash of the side using the global sides
#     """
#     sorted_bytes = np.sort(worker_sides[side].corners).tobytes()
#     return side, hash(sorted_bytes)


def ConnectMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.common.common_progress import ProgressBar
    # from pyhope.common.common_vars import np_mtp
    from pyhope.io.io_vars import MeshFormat, ELEM, ELEMTYPE
    from pyhope.readintools.readintools import GetLogical
    from pyhope.mesh.connect.connect_mortar import ConnectMortar
    # from pyhope.mesh.mesh_common import sidetovol2
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
    nGeo    = mesh_vars.nGeo

    # Native meshio data
    points: Final[np.ndarray] = mesh.points
    cells:  Final[list]       = mesh.cells
    csets:  Final[dict]       = mesh.cell_sets
    cdict:  Final[dict]       = mesh.cells_dict

    bar = ProgressBar(value=len(sides), title='│                 Preparing Sides')

    # Map sides to BC
    # > Create a dict containing only the face corners
    # PERF: The parallel version does not really speed up the process
    # if np_mtp > 0:
    #     # Run in parallel with a chunk size
    #     # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
    #     sides_lst = tuple(s for elem in elems for s in elem.sides)
    #     with Pool(processes=np_mtp, initializer=init_side_worker, initargs=(sides,)) as pool:
    #         results = pool.map(compute_side_hash, sides_lst)
    #     side_corners = dict(results)
    # else:
    #     side_corners = {side: hash(np.sort(sides[side].corners).tobytes()) for elem in elems for side in elem.sides}
    side_corners = {side: hash(np.sort(sides[side].corners).tobytes()) for elem in elems for side in elem.sides}
    # > Create a dict containing only the periodic corners
    peri_corners = {}

    # Build the reverse dictionary
    corner_side = defaultdict(list)
    for side, corners in side_corners.items():
        corner_side[corners].append(side)

    # Set BC and periodic sides
    bcs = mesh_vars.bcs
    vvs = mesh_vars.vvs

    # Find the mapping to the (N-1)-dim elements
    csetMap = { key: tuple(i for i, cell in enumerate(cset) if cell is not None and cast(np.ndarray, cell).size > 0)
                             for key, cset in csets.items()}

    bar.title('│                Processing Sides')
    for key, cset in csets.items():
        # Check if the set is a BC
        bcID = find_bc_index(bcs, key)

        # Ignore the volume zones
        volumeBC = False
        for iMap in csetMap[key]:
            if not any(s in tuple(cdict)[iMap] for s in ['quad', 'triangle']):
                volumeBC = True
                break
        if volumeBC:
            continue

        if bcID is None:
            hopout.error(f'Could not find BC {key} in list, exiting...')

        # Get the list of sides
        for iMap in csetMap[key]:
            # Cache cell types for this mapping to avoid repeated list creation
            cell_types = tuple(cdict)[iMap]
            # Only 2D faces
            if not any(s in cell_types for s in ['quad', 'triangle']):
                continue

            iBCsides = np.array(cset[iMap]).astype(int)
            mapFaces = cells[iMap].data
            # Support for hybrid meshes
            nCorners = 4 if 'quad' in cell_types else 3

            # Map the unique BC sides to our non-unique elem sides
            for iSide in iBCsides:
                # Get the corner nodes
                corners = np.sort(mapFaces[iSide][:nCorners])
                corners = hash(corners.tobytes())

                if corners not in corner_side:
                    print()
                    print(hopout.warn('Malformatted side corners, exiting...'))
                    corners = np.sort(mapFaces[iSide][:nCorners])
                    print(hopout.warn(f'> Side {cell_types}, Nodes {corners}'))
                    for corner in corners:
                        print(hopout.warn('- Coordinates  : [' + ' '.join('{:13.8f}'.format(s) for s in mesh.points[corner]) + ']'))
                    # traceback.print_stack(file=sys.stdout)
                    sys.exit(1)

                # Boundary faces are unique, except for inner/periodic sides
                if len(corner_side[corners]) == 0:
                    continue

                # sideID  = find_keys(face_corners, corners)
                sideIDs = corner_side[corners]
                # Multiple sides with the same corners are only allowed for inner [0,100] and periodic [1] BCs
                match bcs[bcID].type[0]:
                    case 0 | 100:  # Inner side
                        pass
                    case 1:        # Periodic side
                        # Only take the first (positive BC_alpha) side
                        sideIDs = [sideIDs[0]]
                    case _:        # Boundary side
                        # Abort if there are multiple sides with the same corners
                        if len(sideIDs) > 1:
                            hopout.error('Found multiple sides with the same corners, exiting...', traceback=True)

                for sideID in sideIDs:
                    # sides[sideID].update(bcid=bcID)
                    sides[sideID].bcid = bcID

                    # Add the periodic nodes of the periodic sides to the side_corners
                    # > Only negative BC_alpha allowed here
                    if bcs[bcID].type[0] == 1 and bcs[bcID].type[3] > 0:
                        pNodes = np.fromiter((mesh_vars.periNodes[(s, key)] for s in mapFaces[iSide][:nCorners]), dtype=int)
                        pNodes = hash(np.sort(pNodes).tobytes())
                        peri_corners[sideID] = pNodes
                        # Update the reverse dictionary immediately
                        corner_side[pNodes].append(sideID)

                    if bcs[bcID].type[0] != 1:
                        bar.step()

    # Try to connect the inner / periodic sides
    passedTypes = {}
    for val in corner_side.values():
        match len(val):
            case 1:  # BC side
                continue
            case 2:  # Internal side
                sideIDs   = val
                # Flip pyramids
                # if elems[sides[sideIDs[0]].elemID].type % 100 != 5 and \
                #    elems[sides[sideIDs[1]].elemID].type % 100 == 5:
                #     sideIDs   = sideIDs[::-1]

                side0     = sides[sideIDs[0]]
                side1     = sides[sideIDs[1]]
                corners   = side0.corners
                nbcorners = side1.corners

                # Translate to periodic nodes if required
                if side0.bcid is not None and side1.bcid is not None and bcs[side1.bcid].type[0] == 1:
                    nbcorners = np.fromiter((mesh_vars.periNodes[(s, bcs[side1.bcid].name)] for s in side1.corners), dtype=int)

                flipID = flip_analytic(corners[0], nbcorners) + 1

                # Sanity check the flip with the other nodes
                # > INFO: MOVED TO OWN CHECKCONNECT ROUTINE
                # elem   = (elems[side0.elemID], elems[side1.elemID])
                # if elem[0].type % 100 == 8 and elem[1].type % 100 == 8:
                #     # Map the meshio nodes to the tensor-product nodes
                #     elemType = elem[0].type
                #     nodes    = elem[0].nodes[sidetovol2(nGeo, 0     , side0.face, elemType)]
                #     nbNodes  = elem[1].nodes[sidetovol2(nGeo, flipID, side1.face, elemType)]
                #
                #     # Translate to periodic nodes if required
                #     if side0.bcid is not None and side1.bcid is not None and bcs[side1.bcid].type[0] == 1:
                #         nbNodes = np.vectorize(lambda s: mesh_vars.periNodes[(s, bcs[side1.bcid].name)], otypes=[int])(nbNodes)
                #
                #     # Check if the node IDs match
                #     if not np.array_equal(nodes, nbNodes):
                #         # Print the node IDs
                #         print(hopout.warn(f'NodeIDs side[-]: {nodes  }'))
                #         print(hopout.warn(f'NodeIDs side[+]: {nbNodes}'))
                #         hopout.error('Error in connectivity check, node IDs do not match!')

                # Connect the sides
                connect_sides(sideIDs, sides, flipID)
                # Update the progress bar
                bar.step(2)

                # Use guard clauses to avoid unnecessary checks
                if side0.bcid is None:
                    continue     # No boundary condition on first side
                if bcs[side0.bcid].type[0] != 1:
                    continue     # Not a periodic BC on first side
                if bcs[side1.bcid].type[0] != 1:
                    hopout.error('Found internal side with inconsistent BC types, exiting...')
                if not doPeriodicCorrect:
                    continue     # Periodic correction not enabled

                # At this point, we know both sides have periodic BCs
                iVV = bcs[side0.bcid].type[3]
                VV  = vvs[np.abs(iVV) - 1]['Dir'] * np.sign(iVV)
                locSides = tuple(sides[s]        for s in sideIDs)  # noqa: E272
                locElems = tuple(elems[s.elemID] for s in locSides)

                # Only update hexahedral elements
                if any(e.type % 100 != 8 for e in locElems):
                    for e in locElems:
                        passedTypes[e.type] = passedTypes.get(e.type, 0) + 1
                else:
                    periodic_update(locSides, locElems, VV)

            case _:  # Zero or more than 2 sides
                hopout.error('Found internal side with more than two adjacent elements, exiting...', traceback=True)

    if passedTypes:
        print(hopout.warn(hopout.Colors.WARN + '─'*(46-16) + hopout.Colors.END))
        print(hopout.warn('Periodic correction skipped for elements:'))
        print(hopout.warn(hopout.Colors.WARN + '─'*(46-16) + hopout.Colors.END))
        for elemType in ELEM.TYPES:
            if elemType in passedTypes and passedTypes[elemType] > 0:
                print(hopout.warn( ELEMTYPE(elemType) + ': {:12d}'.format(passedTypes[elemType])))

    nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)

    # Mortar sides
    if doMortars:
        # Mortar connections are not supported between mismatching side types
        if any(len(s.corners) != len(nConnSide[0].corners) for s in nConnSide):
            hopout.error('Mortar connections are not supported between mixed side types, exiting...')

        # Connect the mortar sides
        elems, sides = ConnectMortar(nConnSide, nConnCenter, elems, sides, bar)

    nConnSide, nConnCenter = get_nonconnected_sides(sides, mesh)
    if len(nConnSide) > 0:
        print()  # Empty line for spacing
        for side in nConnSide:
            print(hopout.warn(f'> Element {side.elemID+1}, Side {side.face}, Side {side.sideID+1}'))  # noqa: E501
            elem  = elems[side.elemID]
            # nodes = elem.nodes[sidetovol2(nGeo, 0     , side.face, elem.type)]
            nodes = np.transpose(np.array([elem.nodes[s] for s in face_to_nodes(side.face, elem.type, nGeo)]))
            if elem.type % 100 == 8:
                nodes = np.transpose(points[nodes]         , axes=(2, 0, 1))
                print(hopout.warn('- Coordinates  : [' + ' '.join('{:13.8f}'.format(s) for s in nodes[:,  0,  0]) + ']'))
                print(hopout.warn('- Coordinates  : [' + ' '.join('{:13.8f}'.format(s) for s in nodes[:,  0, -1]) + ']'))
                print(hopout.warn('- Coordinates  : [' + ' '.join('{:13.8f}'.format(s) for s in nodes[:, -1,  0]) + ']'))
                print(hopout.warn('- Coordinates  : [' + ' '.join('{:13.8f}'.format(s) for s in nodes[:, -1, -1]) + ']'))
                if side is not nConnSide[-1]:
                    print()  # Empty line for spacing
            else:
                nodes = points[nodes]
                for node in nodes:
                    print(hopout.warn('- Coordinates  : [' + ' '.join('{:13.8f}'.format(s) for s in node) + ']'))
                if side is not nConnSide[-1]:
                    print()  # Empty line for spacing
        hopout.error('Could not connect {} / {} side{}'.format(len(nConnSide), len(sides), '' if len(sides) == 1 else 's'))

    # Close the progress bar
    bar.close()

    # Run garbage collector to release memory
    gc.collect()

    # Count the sides
    nsides             = len(sides)
    sides_conn         = np.empty(nsides, dtype=bool)
    sides_bc           = np.empty(nsides, dtype=bool)
    sides_periodic     = np.empty(nsides, dtype=bool)
    sides_mortar_big   = np.empty(nsides, dtype=bool)
    sides_mortar_small = np.empty(nsides, dtype=bool)

    for i, s in enumerate(sides):
        sides_conn[        i] = s.connection is not None
        sides_bc[          i] = s.bcid       is not None and bcs[s.bcid].type[0] != 100  # noqa: E272
        sides_periodic[    i] = s.bcid       is not None and bcs[s.bcid].type[0] == 1    # noqa: E272
        sides_mortar_big[  i] = s.connection is not None and s.connection < 0
        sides_mortar_small[i] = s.locMortar  is not None                                 # noqa: E272

    # Count each type of side
    ninnersides        = np.sum( sides_conn & ~sides_bc       & ~sides_mortar_small & ~sides_mortar_big)
    nperiodicsides     = np.sum( sides_conn &  sides_periodic & ~sides_mortar_small & ~sides_mortar_big)
    nbcsides           = np.sum(~sides_conn &  sides_bc       & ~sides_mortar_small & ~sides_mortar_big)
    nmortarbigsides    = np.sum(                                                       sides_mortar_big)
    nmortarsmallsides  = np.sum(                                 sides_mortar_small                    )
    nsides             = len(sides) - nmortarsmallsides

    hopout.sep()
    hopout.info(' Number of sides                : {:12d}'.format(nsides))
    hopout.info(' Number of inner sides          : {:12d}'.format(ninnersides))
    hopout.info(' Number of mortar sides (big)   : {:12d}'.format(nmortarbigsides))
    hopout.info(' Number of mortar sides (small) : {:12d}'.format(nmortarsmallsides))
    hopout.info(' Number of boundary sides       : {:12d}'.format(nbcsides))
    hopout.info(' Number of periodic sides       : {:12d}'.format(nperiodicsides))

    mesh_vars.sides = sides
    mesh_vars.elems = elems

    # hopout.sep()
    # hopout.info('CONNECT MESH DONE!')
