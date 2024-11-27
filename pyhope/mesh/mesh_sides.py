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
    from pyhope.mesh.mesh_common import face_to_cgns, faces
    from pyhope.mesh.mesh_vars import ELEM, SIDE
    # ------------------------------------------------------

    mesh   = mesh_vars.mesh
    nElems = 0
    nSides = 0
    sCount = 0
    mesh_vars.elems = []
    mesh_vars.sides = []
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides

    hopout.sep()
    hopout.routine('Eliminating duplicate points')

    # Eliminate duplicate points
    mesh_vars.mesh.points, inverseIndices = np.unique(mesh_vars.mesh.points, axis=0, return_inverse=True)

    # Update the mesh
    for cell in mesh_vars.mesh.cells:
        # Map the old indices to the new ones
        # cell.data = np.vectorize(lambda idx: inverseIndices[idx])(cell.data)
        # Efficiently map all indices in one operation
        cell.data = inverseIndices[cell.data]

    # Loop over all element types
    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        # Get the elements
        ioelems  = mesh.get_cells_type(elemType)
        baseElem = elemType.rstrip(string.digits)
        nIOElems = ioelems.shape[0]
        nIOSides   = mesh_vars.ELEMTYPE.type[baseElem]

        # Create non-unique sides
        mesh_vars.elems.extend([ELEM() for _ in range(nIOElems         )])
        mesh_vars.sides.extend([SIDE() for _ in range(nIOElems*nIOSides)])

        # Create dictionaries
        for iElem in range(nElems, nElems+nIOElems):
            elems[iElem].update(type   = mesh_vars.ELEMMAP(elemType),  # noqa: E251
                                elemID = iElem,                        # noqa: E251
                                sides  = [],                           # noqa: E251
                                nodes  = ioelems[iElem])               # noqa: E251

            # Create the sides
            for iSide in range(nSides, nSides+nIOSides):
                sides[iSide].update(sideType=4)

            # Assign nodes to sides, CGNS format
            for index, face in enumerate(faces(elemType)):
                corners = [ioelems[iElem][s] for s in face_to_cgns(face, elemType)]
                sides[sCount].update(face    = face,                   # noqa: E251
                                     elemID  = iElem,                  # noqa: E251
                                     sideID  = sCount,                 # noqa: E251
                                     locSide = index+1,                # noqa: E251
                                     corners = np.array(corners))      # noqa: E251
                sCount += 1

            # Add to nSides
            nSides += nIOSides

        # Add to nElems
        nElems += nIOElems

    # Append sides to elem
    for iSide, side in enumerate(sides):
        elemID = side.elemID
        sideID = side.sideID
        elems[elemID].sides.append(sideID)
