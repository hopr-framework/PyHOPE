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
import os
# import subprocess
import sys
# import tempfile
# import time
# import traceback
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# import gmsh
import h5py
import meshio
import numpy as np
# import pygmsh
# from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def ReadHOPR(fnames: list, mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_common import LINTEN
    from pyhope.mesh.mesh_common import faces, face_to_cgns
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    hopout.sep()

    # Create an empty meshio object
    points   = mesh.points if len(mesh.points.shape)>1 else np.zeros((0, 3), dtype=np.float64)
    cells    = mesh.cells_dict
    cellsets = mesh.cell_sets

    nodeCoords   = mesh.points
    offsetnNodes = 0
    # offsetnSides = 0
    nSides       = 0

    for fname in fnames:
        # Check if the file is using HDF5 format internally
        if not h5py.is_hdf5(fname):
            hopout.warning('File [󰇘]/{} is not in HDF5 format, exiting...'.format(os.path.basename(fname)))
            sys.exit(1)

        with h5py.File(fname, mode='r') as f:
            # Check if file contains the Hopr version
            if 'HoprVersion' not in f.attrs:
                hopout.warning('File [󰇘]/{} does not contain the Hopr version, exiting...'.format(os.path.basename(fname)))
                sys.exit(1)

            # Read the globalNodeIDs
            nodeInfo   = np.array(f['GlobalNodeIDs'])

            # Read the nodeCoords
            nodeCoords = np.array(f['NodeCoords'])
            indices    = np.unique(nodeInfo, return_index=True)[1]
            nodeCoords = nodeCoords[indices]

            points     = np.append(points, nodeCoords, axis=0)

            # Read nGeo
            nGeo       = cast(int, f.attrs['Ngeo'])
            if nGeo != mesh_vars.nGeo:
                # TODO: FIX THIS WITH A CHANGEBASIS
                filename = os.path.basename(fname)
                hopout.warning('File [󰇘]/{} has different polynomial order than the current mesh, exiting...'.format(filename))
                sys.exit(1)

            # Read the elemInfo and sideInfo
            elemInfo   = np.array(f['ElemInfo'])
            sideInfo   = np.array(f['SideInfo'])
            BCNames    = [s.strip().decode('utf-8') for s in cast(h5py.Dataset, f['BCNames'])]

            # Construct the elements, meshio format
            for elem in elemInfo:
                # Obtain the element type
                elemType = ELEMTYPE.inam[elem[0]]
                if len(elemType) > 1:
                    elemType = elemType[nGeo-2]
                else:
                    elemType = elemType[0]

                linMap    = LINTEN(elem[0], order=nGeo)
                elemNodes = np.arange(elem[4], elem[5])
                elemNodes = np.expand_dims(nodeInfo[elemNodes[linMap]]-1+offsetnNodes, axis=0)

                try:
                    cells[elemType] = np.append(cells[elemType], elemNodes, axis=0)
                except KeyError:
                    cells[elemType] = elemNodes

                # Attach the boundary sides
                # for index, face in enumerate(faces(elemType)):
                sCounter = 0
                for index in range(elem[2], elem[3]):
                    # Account for mortar sides
                    # TODO: Add mortar sides

                    # Obtain the side type
                    sideType  = sideInfo[index, 0]
                    sideBC    = sideInfo[index, 4]

                    BCName    = BCNames[sideBC-1]
                    face      = faces(elemType)[sCounter]
                    corners   = [elemNodes[0][s] for s in face_to_cgns(face, elemType)]

                    # Get the number of corners
                    nCorners  = abs(sideType % 10)
                    sideName  = 'quad' if nCorners == 4 else 'tri'

                    sideNodes = np.expand_dims(corners, axis=0)

                    try:
                        cells[sideName] = np.append(cells[sideName], sideNodes, axis=0)
                    except KeyError:
                        cells[sideName] = sideNodes

                    # Increment the side counter
                    sCounter += 1
                    nSides   += 1

                    if sideBC == 0:
                        continue

                    # Add the side to the cellset
                    # > We did not create any 0D/1D objects, so we do not need to consider any offset
                    try:
                        cellsets[BCName][1] = np.append(cellsets[BCName][1], np.array([nSides-1], dtype=np.uint64))
                    except KeyError:
                        # Pyright does not understand that Meshio expects a list with one None entry
                        cellsets[BCName]    = [None, np.array([nSides-1], dtype=np.uint64)]  # type: ignore

            # Update the offset for the next file
            offsetnNodes += nodeCoords.shape[0]

    mesh   = meshio.Mesh(points    = points,    # noqa: E251
                         cells     = cells,     # noqa: E251
                         cell_sets = cellsets)  # noqa: E251

    return mesh
