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
    import src.mesh.mesh_vars as mesh_vars
    from src.mesh.mesh_common import face_to_cgns, faces
    # ------------------------------------------------------

    mesh   = mesh_vars.mesh
    nElems = 0
    nSides = 0
    sCount = 0
    mesh_vars.elems = []
    mesh_vars.sides = []
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides

    # Loop over all element types
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEM.type.keys()):
            continue

        # Get the elements
        ioelems  = mesh.get_cells_type(elemType)
        baseElem = elemType.rstrip(string.digits)
        nIOElems = ioelems.shape[0]
        nIOSides   = mesh_vars.ELEM.type[baseElem]

        # Create non-unique sides
        mesh_vars.elems.extend([dict() for _ in range(nIOElems         )])
        mesh_vars.sides.extend([dict() for _ in range(nIOElems*nIOSides)])

        # Create dictionaries
        for iElem in range(nElems, nElems+nIOElems):
            elems[iElem]['Type'  ] = mesh_vars.ELEMMAP(elemType)
            elems[iElem]['ElemID'] = iElem
            elems[iElem]['Sides' ] = []
            elems[iElem]['Nodes' ] = ioelems[iElem]

            # Create the sides
            for iSide in range(nSides, nSides+nIOSides):
                sides[iSide]['Type'  ] = 4  # FIXME: THIS NEEDS TREATMENT FOR NON-HEXAS

            # Assign nodes to sides, CGNS format
            for index, face in enumerate(faces()):
                corners = [ioelems[iElem][s] for s in face_to_cgns(face)]
                sides[sCount].update({'ElemID' : iElem})
                sides[sCount].update({'SideID' : sCount})
                sides[sCount].update({'LocSide': index+1})
                sides[sCount].update({'Corners': np.array(corners)})
                sCount += 1

            # Add to nSides
            nSides += nIOSides

        # Add to nElems
        nElems += nIOElems

    # Append sides to elem
    for iSide, side in enumerate(sides):
        elemID = side['ElemID']
        sideID = side['SideID']
        elems[elemID]['Sides'].append(sideID)
