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
import gc
from typing import cast, final
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
    from pyhope.common.common_progress import ProgressBar
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    import numpy as np
    # INFO: Alternative Hilbert curve sorting (not on PyPI)
    # from hilsort import hilbert_sort
    # ------------------------------------------------------

    hopout.routine('Sorting elements along space-filling curve')

    mesh  = mesh_vars.mesh
    elems = mesh_vars.elems
    sides = mesh_vars.sides

    totalElems = len(elems)
    bar = ProgressBar(value=totalElems, title='│              Preparing Elements', length=33)

    # Global bounding box
    points = mesh.points
    xmin = np.min(points, axis=0)
    xmax = np.max(points, axis=0)

    # Calculate the element barycenters and associated element offsets
    elem_bary  = calc_elem_bary(elems)

    # Calculate the space-filling curve resolution for the given KIND
    kind = 4
    nbits, spacing = SFCResolution(kind, xmin, xmax)

    # Discretize the element positions according to the chosen resolution
    elem_disc      = Coords2Int(elem_bary, spacing, xmin)

    # Generate the space-filling curve and order elements along it
    hc             = HilbertCurve(p=nbits, n=3, n_procs=np_mtp)

    distances      = np.array(hc.distances_from_points(elem_disc))  # bottleneck
    sorted_indices = np.argsort(distances)

    # INFO: Alternative Hilbert curve sorting (not on PyPI)
    # distances      = np.array(hilbert_sort(8, elem_bary))
    # Find the new sorting with the old elem_bary
    # value_to_index = {tuple(value.tolist()): idx for idx, value in enumerate(distances)}
    # Now, create an array that maps each element to the new sorting
    # sorted_indices = np.array([value_to_index[tuple(val.tolist())] for val in elem_bary])

    # Initialize sorted cells
    sorted_elems   = tuple(elems[i] for i in sorted_indices)
    sorted_sides   = []

    bar.title('│             Processing Elements')

    # Overwrite the elem/side IDs
    offsetSide = 0
    for elemID, elem in enumerate(sorted_elems):
        elem.elemID = elemID

        for key, val in enumerate(elem.sides):
            side        = sides[val]
            side.sideID = offsetSide + key
            side.elemID = elemID

            sorted_sides.append(side)

        # Correct the sideID
        nSides      = len(elem.sides)
        elem.sides  = list(range(offsetSide, offsetSide + nSides))
        offsetSide += nSides
        bar.step()

    mesh_vars.elems = sorted_elems
    mesh_vars.sides = sorted_sides

    # Close the progress bar
    bar.close()


def SortMeshByIJK() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_common import count_elems, calc_elem_bary
    from pyhope.common.common_progress import ProgressBar
    # ------------------------------------------------------

    hopout.routine('Sorting elements along I,J,K direction')

    mesh  = mesh_vars.mesh
    elems = mesh_vars.elems
    sides = mesh_vars.sides

    # Calculate the element bary centers and type offsets
    elemBary = calc_elem_bary(elems)

    # Calculate bounding box and conversion factor
    ptp_elemBary = np.ptp(elemBary, axis=0)
    lower        = np.min(np.min(elemBary, axis=0) - 0.1 * np.min(ptp_elemBary, axis=0), axis=0)
    upper        = np.max(np.max(elemBary, axis=0) + 0.1 * np.max(ptp_elemBary, axis=0), axis=0)

    # Convert coordinates to integer space
    box       = tBox(np.floor(lower), np.ceil(upper))
    intCoords = np.rint((elemBary - box.mini) * box.spacing).astype(int)

    # Initialize lists
    nElems    = count_elems(mesh)
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
        nElemsIJK = np.array((nElems, 1, 1))
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
        tIJK = np.copy(nElemsIJK)
        nElemsIJK[0] = round(np.sqrt(tIJK[1] * tIJK[2] / tIJK[0]))
        nElemsIJK[1] = round(np.sqrt(tIJK[0] * tIJK[2] / tIJK[1]))
        nElemsIJK[2] = round(np.sqrt(tIJK[0] * tIJK[1] / tIJK[2]))

    # Check for consistency in the number of elements
    if np.prod(nElemsIJK) != nElems:
        hopout.warning('Problem during sort elements by coordinate: nElems /= nElems_I * Elems_J * nElems_K')

    hopout.sep()
    hopout.info(' Number of structured dirs      : {}'.format(nStructDirs))
    hopout.info(' Number of elems [I,J,K]        : {}'.format(nElemsIJK))

    totalElems = len(elems)
    bar = ProgressBar(value=totalElems, title='│              Preparing Elements', length=33)

    # Now sort the elements based on z, y, then x coordinates
    intList        = (intCoords[:, 2] * 10000 + intCoords[:, 1]) * 10000 + intCoords[:, 0]
    sorted_indices = np.argsort(intList)

    # Initialize sorted cells
    sorted_elems   = [elems[i] for i in sorted_indices]
    sorted_sides   = []

    bar.title('│             Processing Elements')

    # Overwrite the elem/side IDs
    offsetSide = 0
    for elemID, elem in enumerate(sorted_elems):
        elem.elemID = elemID

        for key, val in enumerate(elem.sides):
            side        = sides[val]
            side.sideID = offsetSide + key
            side.elemID = elemID

            sorted_sides.append(side)

        # Correct the sideID
        nSides      = len(elem.sides)
        elem.sides  = list(range(offsetSide, offsetSide + nSides))
        offsetSide += nSides
        bar.step()

    mesh_vars.elems = sorted_elems
    mesh_vars.sides = sorted_sides

    bar.close()


def SortMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('SORT MESH...')
    hopout.sep()

    mesh_vars.sortIJK = GetLogical('doSortIJK')
    hopout.sep()

    # Sort the mesh
    if mesh_vars.sortIJK:
        SortMeshByIJK()
    else:
        SortMeshBySFC()

    # Run garbage collector to release memory
    gc.collect()
