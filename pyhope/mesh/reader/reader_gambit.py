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
# import itertools
# import os
# import shutil
# import sys
# import tempfile
# from dataclasses import dataclass, field
# from functools import cache
# from string import digits
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# import h5py
import meshio
import numpy as np
# from alive_progress import alive_bar
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def lines_that_equal(     string, fp, start_idx=0) -> list[int]:
    return [num for num, line in enumerate(fp[start_idx:]) if line.strip() == string]


def lines_that_contain(   string, fp, start_idx=0) -> list[int]:
    return [num for num, line in enumerate(fp[start_idx:], start=start_idx) if string in line]


def lines_that_start_with(string, fp, start_idx=0) -> list[int]:
    return [num for num, line in enumerate(fp[start_idx:]) if line.startswith(string)]


def lines_that_end_with(string, fp, start_idx=0) -> list[int]:
    return [num for num, line in enumerate(fp[start_idx:]) if line.rstrip().endswith(string)]


def ReadGambit(fnames: list, mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # import pyhope.mesh.mesh_vars as mesh_vars
    # from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_3D
    # from pyhope.mesh.mesh_common import LINTEN
    # from pyhope.mesh.mesh_common import faces, face_to_nodes
    # from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    hopout.sep()

    # Create an empty meshio object
    points   = mesh.points if len(mesh.points.shape)>1 else np.zeros((0, 3), dtype=np.float64)
    pointl   = cast(list, points.tolist())
    cells    = mesh.cells_dict
    # cellsets = {}

    nodeCoords   = mesh.points
    # offsetnNodes = nodeCoords.shape[0]
    # nSides       = np.zeros(2, dtype=int)

    # Instantiate ELEMTYPE
    # elemTypeClass = ELEMTYPE()

    for fname in fnames:
        # Check if the file is using ASCII format internally
        with open(fname, 'r') as f:
            # Check if the file is in ASCII format
            try:
                # Read the file content
                content   = f.readlines()
                useBinary = not any('CONTROL INFO' in line for line in content)
            except UnicodeDecodeError:
                content   = None  # FIXME
                useBinary = True

            # Cache the mapping here, so we consider the mesh order
            # linCache   = {}

            if not useBinary:
                # Search for the line containing the number of elements
                elemLine = lines_that_contain('NUMNP'            , content)[0] + 1
                # Read and unpack the number of elements
                npoints, nelems, _, nbcs, _, _ = map(int, content[elemLine].strip().split())

                # Search for the line starting the node coordinates
                nodeLine = lines_that_contain('NODAL COORDINATES', content)[0]

                # Iterate and unpack the node coordinates
                nodeCoords = content[nodeLine+1:nodeLine+npoints+1]
                nodeCoords = np.genfromtxt(nodeCoords, dtype=np.float64, delimiter=None, usecols=(1, 2, 3))
                pointl.extend(nodeCoords)

                # Search for the line starting the element connectivity
                elemLine = lines_that_contain('ELEMENTS/CELLS', content)[0] + 1
                elemIter = iter(content[elemLine:])

                # Iterate and unpack the element connectivity
                for line in elemIter:
                    if 'ENDOFSECTION' in line:
                        break

                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    try:
                        elemID, gType, nNodes, *elemNodes = tokens
                        elemID, gType, nNodes             = int(elemID), int(gType), int(nNodes)
                    except ValueError:
                        continue

                    # Keep extending the element connectivity until elemNodes is reached
                    while len(elemNodes) < nNodes:
                        elemNodes.extend(next(elemIter).strip().split())

                    # TODO: Map gType to the meshio cell type
                    elemType = 'hexahedron'
                    # elemNum = 108

                    # ChangeBasis currently only supported for hexahedrons
                    # if elemNum in linCache:
                    #     mapLin = linCache[elemNum]
                    # else:
                    #     _, mapLin = LINTEN(elemNum, order=mesh_vars.nGeo)
                    #     mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))
                    #     linCache[elemNum] = mapLin
                    mapLin = np.asarray([0, 1, 5, 4, 2, 3, 7, 6])

                    # Convert elemNodes to a numpy array of integers
                    elemNodes = np.array(elemNodes, dtype=np.uint64)
                    elemNodes = elemNodes[mapLin] - 1

                    if elemType in cells:
                        cells[elemType].append(elemNodes.astype(np.uint64))
                    else:
                        cells[elemType] = [elemNodes.astype(np.uint64)]

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    # > CS4: We create the final meshio.Mesh object with cell_sets
    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = cells)      # noqa: E251
                         # cell_sets = cell_sets)  # noqa: E251

    # Run garbage collector to release memory
    gc.collect()

    return mesh
