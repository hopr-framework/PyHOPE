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
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import traceback
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
ELEMTYPE = 'hexahedron'
# ==================================================================================================================================


def flip(side, nbside):
    ''' Determines the flip of the side-to-side connection
        flip = 1 : 1st node of neighbor side = 1st node of side
        flip = 2 : 2nd node of neighbor side = 1st node of side
        flip = 3 : 3rd node of neighbor side = 1st node of side
        flip = 4 : 4th node of neighbor side = 1st node of side
    '''
    # Local imports ----------------------------------------
    from src.common.common import find_index
    # ------------------------------------------------------
    return find_index(nbside, side[0])


def ConnectMesh():
    # Local imports ----------------------------------------
    from src.io.io_vars import MeshFormat
    from src.common.common import find_key, find_keys
    import src.io.io_vars as io_vars
    from src.mesh.mesh_common import faces, face_to_corner
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    # ------------------------------------------------------

    match io_vars.outputformat:
        case MeshFormat.FORMAT_HDF5:
            pass
        case _:
            return

    hopout.separator()
    hopout.info('CONNECT MESH...')

    mesh = mesh_vars.mesh

    # Create non-unique sides
    ioelems = mesh.get_cells_type(ELEMTYPE)
    nElems  = ioelems.shape[0]
    mesh_vars.elems = [None] * nElems
    mesh_vars.sides = [None] * nElems * 6
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides
    quads   = mesh.get_cells_type('quad')

    # Create dictionaries
    for iElem, elem in enumerate(elems):
        elems[iElem] = dict()
        elems[iElem]['Type'  ] = 108  # FIXME
        elems[iElem]['ElemID'] = iElem
        elems[iElem]['Sides' ] = []
        elems[iElem]['Nodes' ] = ioelems[iElem]

    for iSide, side in enumerate(sides):
        sides[iSide] = dict()
        sides[iSide]['Type'  ] = 104  # FIXME

    # Assign nodes to sides, CGNS format
    count = 0
    for iElem in range(nElems):
        for index, face in enumerate(faces()):
            corners = [ioelems[iElem][s] for s in face_to_corner(face)]
            sides[count].update({'ElemID' : iElem})
            sides[count].update({'SideID' : count})
            sides[count].update({'Corners': np.array(corners)})
            count += 1

    # Append sides to elem
    for iSide, side in enumerate(sides):
        elemID = side['ElemID']
        sideID = side['SideID']
        elems[elemID]['Sides'].append(sideID)

    # cell_sets contain the face IDs [dim=2]
    # > Offset is calculated with entities from [dim=0, dim=1]
    offsetcs =  len(mesh.get_cells_type('vertex'))
    offsetcs += len(mesh.get_cells_type('line'))

    # Map sides to BC
    # > Create a dict containing only the face corners
    side_corners = dict()
    for iSide, side in enumerate(sides):
        corners = np.sort(side['Corners'])
        corners = hash(corners.tostring())
        side_corners.update({iSide: corners})

    # Build the reverse dictionary
    corner_side = dict()
    for key, val in side_corners.items():
        if val not in corner_side:
            corner_side[val] = [key]
        else:
            corner_side[val].append(key)

    for key, cset in mesh.cell_sets.items():
        # Check if the set is a BC
        if key:
            # Get the BCIndex from the list
            for iBC, bc in enumerate(mesh_vars.bcs):
                if key in bc['Name']:
                    bcid = iBC

            # Get the list of sides
            iBCsides = cset[1] - offsetcs

            # Map the unique quad sides to our non-unique elem sides
            for iSide in iBCsides:
                # Get the quad corner nodes
                corners = np.sort(np.array(quads[iSide]))
                corners = hash(corners.tostring())

                # Boundary faces are unique
                # sideID  = find_key(face_corners, corners)
                sideID = corner_side[corners][0]
                sides[sideID].update({'BCID': bcid})

    # Try to connect the inner / periodic sides
    for iSide, side in enumerate(sides):
        if 'BCID' not in side:
            # Check if side is already connected
            if 'Connection' not in side:
                corners = side_corners[iSide]
                # Find the matching sides
                # sideIDs  = find_keys(face_corners, corners)
                sideIDs = corner_side[corners]
                # Sanity check
                if len(sideIDs) != 2:
                    hopout.warning('Found internal side with more than two adjacent elements, exiting...')
                    traceback.print_stack(file=sys.stdout)
                    sys.exit()

                # Connect the sides
                sides[sideIDs[0]].update({'Connection': sideIDs[1]})
                sides[sideIDs[0]].update({'Flip'      : flip(sides[sideIDs[0]]['Corners'], sides[sideIDs[1]]['Corners']) + 1})
                sides[sideIDs[1]].update({'Connection': sideIDs[0]})
                sides[sideIDs[1]].update({'Flip'      : flip(sides[sideIDs[1]]['Corners'], sides[sideIDs[0]]['Corners']) + 1})


    # Count the sides
    nsides         = len(sides)
    ninnersides    = 0
    nbcsides       = 0
    nperiodicsides = 0
    for iSide, side in enumerate(sides):
        if 'BCID'       in side:
            nbcsides    += 1
        if 'Connection' in side:
            ninnersides += 1

    hopout.sep()
    hopout.info('Number of sides          : {:9d}'.format(nsides))
    hopout.info('Number of inner sides    : {:9d}'.format(ninnersides))
    hopout.info('Number of boundary sides : {:9d}'.format(nbcsides))
    hopout.info('Number of periodic sides : {:9d}'.format(nperiodicsides))
    hopout.sep()

    # Connect the remaining sides
    # TODO: PERIODIC!

    hopout.info('CONNECT MESH DONE!')
