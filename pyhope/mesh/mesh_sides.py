#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024-2026 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
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
    # INFO: Tuples should be faster than lists but cumbersome to add elements
    # mesh_vars.elems = ()
    # mesh_vars.sides = ()
    mesh_vars.elems = []
    mesh_vars.sides = []
    elems   = mesh_vars.elems
    sides   = mesh_vars.sides

    totalElems = 0
    for elemType in mesh.cells_dict:
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type):
            continue

        totalElems += mesh.get_cells_type(elemType).shape[0]

    # Use a moderate chunk size to bound intermediate progress updates
    chunk = max(1, min(1000, max(10, int(totalElems/(400)))))
    bar = ProgressBar(value=totalElems, title='│             Processing Elements', length=33, chunk=chunk)

    # Loop over all element types
    for elemType in mesh.cells_dict:
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type):
            continue

        # Get the elements
        ioelems   = mesh.get_cells_type(elemType)
        elemMap   = mesh_vars.ELEMMAP(elemType)
        nIOElems  = ioelems.shape[0]
        elemFaces = faces(elemType)
        nIOSides  = len(elemFaces)

        # Map volume cell sets to elements
        iocsets = mesh.cell_sets_dict
        elemSet: list[int | None]  = [None for _ in range(nIOElems)]

        for key, val in iocsets.items():
            if elemType not in val:
                continue

            # Extract the zoneID
            if key.isdigit():
                zoneID  = key
            else:
                # Get a list of the valid zone names
                zoneIDs = [k for k, v in iocsets.items() if set(v.keys()).issubset(set(mesh_vars.ELEMTYPE.name.keys()))]
                # Build a dictionary mapping each zone name to an index
                zoneIDs = {name: i for i, name in enumerate(zoneIDs, 1)}
                zoneID  = zoneIDs[key]

            for elemID in val[elemType]:
                elemSet[elemID] = zoneID

        # Create non-unique sides
        # mesh_vars.elems += tuple(ELEM() for _ in range(nIOElems         ))
        # mesh_vars.sides += tuple(SIDE() for _ in range(nIOElems*nIOSides))
        # elems = mesh_vars.elems
        # sides = mesh_vars.sides
        elems.extend([ELEM() for _ in range(nIOElems         )])  # pyright: ignore[reportArgumentType]
        sides.extend([SIDE() for _ in range(nIOElems*nIOSides)])  # pyright: ignore[reportArgumentType]

        sideIDs = nSides + np.arange(nIOElems*nIOSides).reshape(nIOElems, nIOSides)

        # Create the corner faces
        corner_faces  = tuple(face_to_cgns(s, elemType) for s in elemFaces)        # noqa: E272
        corner_length = tuple(len(c)                    for c in corner_faces)     # noqa: E272
        corner_index  = tuple(np.array(c, dtype=int)    for c in corner_faces)     # noqa: E272
        corner_coll   = tuple(ioelems[:, c]             for c in corner_index)     # noqa: E272

        # Create dictionaries
        for e in range(nIOElems):
            iElem = nElems + e
            elem  = elems[iElem]

            # elem.update(type   = elemMap,                      # noqa: E251
            #             elemID = iElem,                        # noqa: E251
            #             sides  = [],                           # noqa: E251
            #             nodes  = ioelems[iElem])               # noqa: E251
            elem.type   = elemMap                       # noqa: E251
            elem.elemID = iElem                         # noqa: E251
            elem.sides  = sideIDs[e]                    # noqa: E251
            elem.nodes  = ioelems[e]                    # noqa: E251

            # Create the zone
            # > Account for different elemTypes in the index
            zone = elemSet[e]
            if zone is not None:
                elem.zone = zone

            # Create the sides and assign the corners to the sides, CGNS format
            sCount = nSides + e*nIOSides
            for index in range(nIOSides):
                side          = sides[sCount + index]
                side.sideType = corner_length[index]        # noqa: E251
                side.face     = elemFaces[index]            # noqa: E251
                side.elemID   = iElem                       # noqa: E251
                side.sideID   = sCount + index              # noqa: E251
                side.locSide  = index + 1                   # noqa: E251
                side.corners  = corner_coll[index][e]        # noqa: E251

            # Update progress bar
            bar.step()

        # Add to nSides and nElems
        nSides += nIOElems*nIOSides
        nElems += nIOElems

    # Close the progress bar
    bar.close()
