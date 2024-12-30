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
from typing import cast, final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def Coords2Int(coords: np.ndarray, spacing: np.ndarray, xmin: np.ndarray) -> np.ndarray:
    """ Compute the integer discretization in each direction
    """
    disc = np.round((coords - xmin) * spacing)
    return np.asarray(disc)


def SFCResolution(kind: int, xmin: np.ndarray, xmax: np.ndarray) -> tuple[int, np.ndarray]:
    """ Compute the resolution of the SFC for the given bounding box
        and the given integer kind
    """
    blen    = xmax - xmin
    nbits   = (kind*8 - 1)  # / 3.
    intfact = cast(int, 2**nbits-1)
    spacing = np.ceil(intfact/blen)

    return np.ceil(nbits).astype(int), spacing


@final
class tBox:
    def __init__(self, mini: int, maxi: int):
        self.mini = mini
        self.intfact = 0
        self.spacing = np.zeros(3)
        self._set_bounding_box(mini, maxi)

    def _set_bounding_box(self, mini, maxi):
        blen = maxi - mini
        nbits = (np.iinfo(np.int64).bits - 1) // 3
        self.intfact = 2 ** nbits - 1
        if np.all(blen > 0):
            self.spacing = self.intfact / blen
        else:
            self.spacing = self.intfact


def SortMeshBySFC() -> None:
    # Local imports ----------------------------------------
    from hilbertcurve.hilbertcurve import HilbertCurve
    from pyhope.common.common_vars import np_mtp
    from pyhope.mesh.mesh_common import calc_elem_bary
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    import meshio
    import numpy as np
    # ------------------------------------------------------

    hopout.routine('Sorting elements along space-filling curve')

    mesh = mesh_vars.mesh

    # Global bounding box
    points = mesh.points
    xmin = np.min(points, axis=0)
    xmax = np.max(points, axis=0)

    # Calculate the element barycenters and associated element offsets
    elem_bary, type_offsets = calc_elem_bary(mesh)

    # Calculate the space-filling curve resolution for the given KIND
    kind = 4
    nbits, spacing = SFCResolution(kind, xmin, xmax)

    # Discretize the element positions according to the chosen resolution
    elem_disc = Coords2Int(elem_bary, spacing, xmin)

    # Generate the space-filling curve and order elements along it
    hc             = HilbertCurve(p=nbits, n=3, n_procs=np_mtp)
    distances      = np.array(hc.distances_from_points(elem_disc))  # bottleneck
    sorted_indices = np.argsort(distances)

    # Initialize sorted cells for each type
    sorted_cells = []

    for cellType, offset in zip(mesh.cells, type_offsets):
        if offset is not None:  # 3D element type
            # Get the number of elements for this type
            count = cellType.data.shape[0]

            # Filter global indices belonging to this type
            type_indices  = sorted_indices[(sorted_indices >= offset) & (sorted_indices < offset + count)]

            # Map global indices to local indices for this type
            local_indices = type_indices - offset

            # Sort the data for this cell type
            sorted_data = cellType.data[local_indices]
            sorted_cells.append(meshio.CellBlock(cellType.type, sorted_data))
        else:
            # Non-3D elements remain unsorted
            sorted_cells.append(cellType)

    # Overwrite the old mesh
    mesh_vars.mesh = meshio.Mesh(points=points,
                                 cells=sorted_cells,
                                 cell_sets=mesh.cell_sets)


def SortMeshByIJK():
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_common import count_elems, calc_elem_bary
    # ------------------------------------------------------

    hopout.routine('Sorting elements along I,J,K direction')

    # We only need the volume cells
    mesh        = mesh_vars.mesh
    nElems      = count_elems(mesh)
    # Calculate the element bary centers and type offsets
    elemBary, type_offsets = calc_elem_bary(mesh)

    # Calculate bounding box and conversion factor
    ptp_elemBary = np.ptp(elemBary, axis=0)
    lower        = np.min(elemBary, axis=0) - 0.1 * ptp_elemBary
    upper        = np.max(elemBary, axis=0) + 0.1 * ptp_elemBary
    box = tBox(lower, upper)

    # Convert coordinates to integer space
    intCoords = np.rint((elemBary - box.mini) * box.spacing).astype(int)

    # Initialize lists
    nElemsIJK = np.zeros(3, dtype=int)
    structDir = np.zeros(3, dtype=bool)
    tol       = 1
    intList   = []

    for dir in range(3):
        # Sort elements by coordinate directions
        intList = intCoords[:, dir]
        sortedIndices = np.argsort(intList)
        intListSorted = intList[sortedIndices]

        # Determine structured directions
        nElems_min, nElems_max = nElems, 0
        counter = 1

        # Count the consecutive matching values to determine structure
        for iElem in range(1, nElems):
            if abs(intListSorted[iElem] - intListSorted[iElem - 1]) > tol:
                nElems_min = min(nElems_min, counter)
                nElems_max = max(nElems_max, counter)
                counter = 1
            else:
                counter += 1

        if nElems_max != nElems_min:
            nElemsIJK[dir] = 0  # Not structured
            structDir[dir] = False
        else:
            nElemsIJK[dir] = nElems_max
            structDir[dir] = True

    nStructDirs = np.sum(structDir)

    # Adjust nElemsIJK based on structured directions
    if nStructDirs == 0:
        nElemsIJK = np.array([nElems, 1, 1])
    elif nStructDirs == 1:
        structured_dir = np.argmax(structDir)
        nElemsIJK[structured_dir] = nElems // nElemsIJK[structured_dir]
        nElemsIJK[(structured_dir + 1) % 3] = nElems // nElemsIJK[structured_dir]
        nElemsIJK[(structured_dir + 2) % 3] = 1
    elif nStructDirs == 2:
        non_structured_dir = np.argmin(structDir)
        nElemsIJK[non_structured_dir] = 1
        nElemsIJK[~structDir] = nElemsIJK[~structDir][::-1]
    else:
        nElemsIJK[0] = int(np.sqrt(nElemsIJK[1] * nElemsIJK[2] / nElemsIJK[0]))
        nElemsIJK[1:3] = nElemsIJK[2:4] // nElemsIJK[0]

    # Check for consistency in the number of elements
    if np.prod(nElemsIJK) != nElems:
        hopout.warning('Problem during sort elements by coordinate: nElems /= nElems_I * Elems_J * nElems_K')

    hopout.sep()
    hopout.info(' Number of structured dirs: {}'.format(nStructDirs))
    hopout.info(' Number of elems [I,J,K]  : {}'.format(nElemsIJK))
    hopout.separator()

    # Now sort the elements based on z, y, then x coordinates
    intList        = (intCoords[:, 2] * 10000 + intCoords[:, 1]) * 10000 + intCoords[:, 0]
    sorted_indices = np.argsort(intList)

    # Create a new mesh with only volume elements and sorted along SFC
    points   = mesh_vars.mesh.points
    cells    = mesh_vars.mesh.cells
    cellsets = mesh_vars.mesh.cell_sets

    sorted_cells = []
    for cellType, offset in zip(mesh.cells, type_offsets):
        if offset is not None:  # 3D element type
            # Get the number of elements for this type
            count = cellType.data.shape[0]

            # Filter global indices belonging to this type
            type_indices  = sorted_indices[(sorted_indices >= offset) & (sorted_indices < offset + count)]

            # Map global indices to local indices for this type
            local_indices = type_indices - offset

            # Sort the data for this cell type
            sorted_data = cellType.data[local_indices]
            sorted_cells.append(meshio.CellBlock(cellType.type, sorted_data))
        else:
            # Non-3D elements remain unsorted
            sorted_cells.append(cellType)

    # Overwrite the old mesh
    mesh_vars.mesh   = meshio.Mesh(points=points,
                                   cells=cells,
                                   cell_sets=cellsets)


def SortMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.separator()
    mesh_vars.sortIJK = GetLogical('doSortIJK')
    hopout.sep()

    # Sort the mesh
    if mesh_vars.sortIJK:
        SortMeshByIJK()
    else:
        SortMeshBySFC()
