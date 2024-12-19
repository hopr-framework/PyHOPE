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
import string
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


def GenerateSides() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_progress import ProgressBar
    from pyhope.mesh.mesh_common import faces, face_to_cgns
    from pyhope.mesh.mesh_vars import ELEM, SIDE
    # ------------------------------------------------------

    hopout.sep()
    hopout.routine('Generating sides')

    mesh   = mesh_vars.mesh
    nElems = 0
    nSides = 0
    sCount = 0
    # INFO: Tuples should be faster than lists but cumbersome to add elements
    # mesh_vars.elems = ()
    # mesh_vars.sides = ()
    mesh_vars.elems = []
    mesh_vars.sides = []
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides

    totalElems = 0
    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        totalElems += mesh.get_cells_type(elemType).shape[0]

    bar = ProgressBar(value=totalElems, title='│             Processing Elements', length=33)

    # Loop over all element types
    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        # Get the elements
        ioelems  = mesh.get_cells_type(elemType)
        elemMap  = mesh_vars.ELEMMAP(elemType)
        nIOElems = ioelems.shape[0]
        nIOSides = mesh_vars.ELEMTYPE.type[elemType.rstrip(string.digits)]

        # Create non-unique sides
        # mesh_vars.elems += tuple(ELEM() for _ in range(nIOElems         ))
        # mesh_vars.sides += tuple(SIDE() for _ in range(nIOElems*nIOSides))
        # elems = mesh_vars.elems
        # sides = mesh_vars.sides
        mesh_vars.elems.extend([ELEM() for _ in range(nIOElems         )])
        mesh_vars.sides.extend([SIDE() for _ in range(nIOElems*nIOSides)])

        # Create the corner faces
        corner_faces = [face_to_cgns(s, elemType) for s in faces(elemType)]

        # Create dictionaries
        for iElem in range(nElems, nElems+nIOElems):
            # elems[iElem].update(type   = elemMap,                      # noqa: E251
            #                     elemID = iElem,                        # noqa: E251
            #                     sides  = [],                           # noqa: E251
            #                     nodes  = ioelems[iElem])               # noqa: E251
            elems[iElem].type   = elemMap                       # noqa: E251
            elems[iElem].elemID = iElem                         # noqa: E251
            elems[iElem].sides  = []                            # noqa: E251
            elems[iElem].nodes  = ioelems[iElem]                # noqa: E251

            # Create the sides
            for iSide in range(nSides, nSides+nIOSides):
                # sides[iSide].update(sideType=4)
                sides[iSide].sideType=4

            # Assign corners to sides, CGNS format
            for index, face in enumerate(faces(elemType)):
                corners = [ioelems[iElem][s] for s in corner_faces[index]]
                # sides[sCount].update(face    = face,                   # noqa: E251
                #                      elemID  = iElem,                  # noqa: E251
                #                      sideID  = sCount,                 # noqa: E251
                #                      locSide = index+1,                # noqa: E251
                #                      corners = np.array(corners))      # noqa: E251
                sides[sCount].face    = face                    # noqa: E251
                sides[sCount].elemID  = iElem                   # noqa: E251
                sides[sCount].sideID  = sCount                  # noqa: E251
                sides[sCount].locSide = index+1                 # noqa: E251
                sides[sCount].corners = np.array(corners)      # noqa: E251
                sCount += 1

            # Add to nSides
            nSides += nIOSides

            # Update progress bar
            bar.step()

        # Add to nElems
        nElems += nIOElems

    # Append sides to elem
    for iSide, side in enumerate(sides):
        elemID = side.elemID
        sideID = side.sideID
        elems[elemID].sides.append(sideID)
