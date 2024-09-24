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
import sys
from typing import Tuple
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
    return disc


def centeroidnp(coords: np.ndarray) -> np.ndarray:
    """ Compute the centroid (barycenter) of a set of coordinates
    """
    length = coords.shape[0]
    sum_x  = np.sum(coords[:, 0])
    sum_y  = np.sum(coords[:, 1])
    sum_z  = np.sum(coords[:, 2])
    return np.array([sum_x/length, sum_y/length, sum_z/length])


def SFCResolution(kind: int, xmin: np.ndarray, xmax: np.ndarray) -> Tuple[int, np.ndarray]:
    """ Compute the resolution of the SFC for the given bounding box
        and the given integer kind
    """
    blen    = xmax - xmin
    nbits   = (kind*8 - 1) / 3.
    intfact = 2**nbits-1
    spacing = np.ceil(intfact/blen)

    return np.ceil(nbits).astype(int), spacing


def SortMesh() -> None:
    # Local imports ----------------------------------------
    from hilbertcurve.hilbertcurve import HilbertCurve
    from src.common.common_vars import np_mtp
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    # ------------------------------------------------------

    hopout.separator()
    hopout.routine('Sorting elements along space-filling curve')

    huge = sys.float_info.max
    xmin = np.array([ huge,  huge,  huge])
    xmax = np.array([-huge, -huge, -huge])
    # We only need the volume cells
    mesh   = mesh_vars.mesh
    nElems = 0
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEM.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)
        nElems += ioelems.shape[0]

        for cell in ioelems:
            for point in mesh_vars.mesh.points[cell]:
                xmin = np.minimum(xmin, point)
                xmax = np.maximum(xmax, point)

    # Calculate the element bary centers
    elemBary = [np.ndarray(3)] * nElems
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEM.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)

        for elemID, cell in enumerate(ioelems):
            elemBary[elemID] = centeroidnp(mesh_vars.mesh.points[cell])

    # Calculate the space-filling curve resolution for the given KIND
    kind = 4
    nbits, spacing = SFCResolution(kind, xmin, xmax)

    # Discretize the element positions along according to the chosen resolution
    elemDisc = [np.ndarray(3)] * nElems
    for iType, elemType in enumerate(mesh.cells_dict.keys()):
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEM.type.keys()):
            continue

        ioelems = mesh.get_cells_type(elemType)

        for elemID, cell in enumerate(ioelems):
            elemDisc[elemID] = Coords2Int(elemBary[elemID], spacing, xmin)

    # Generate the space-filling curve and order elements along it
    hc = HilbertCurve(p=nbits, n=3, n_procs=np_mtp)
    distances = hc.distances_from_points(elemDisc)

    # Create a new mesh with only volume elements and sorted along SFC
    points   = mesh_vars.mesh.points
    cells    = mesh_vars.mesh.cells
    cellsets = mesh_vars.mesh.cell_sets

    for iCell, cellType in enumerate(cells):
        if any(s in cellType.type for s in mesh_vars.ELEM.type.keys()):
            # FIXME: THIS BREAKS FOR HYBRID MESHES SINCE THE LIST ARE NOT THE SAME LENGTH THEN!
            cellType.data = np.asarray([x.tolist() for _, x in sorted(zip(distances, cellType.data))])

    # Overwrite the old mesh
    mesh   = meshio.Mesh(points=points,
                         cells=cells,
                         cell_sets=cellsets)

    mesh_vars.mesh = mesh
