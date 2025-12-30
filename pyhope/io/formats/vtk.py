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
from functools import cache
from typing import Tuple
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


@cache                                                       # pragma: no cover
def HEXMAPVTK(order: int) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """ VTK -> IJK ordering for high-order hexahedrons
        > Loosely based on [Gmsh] "generatePointsHexCGNS"
        > [Jens Ulrich Kreber] "paraview-scripts/node_ordering. py"

        > The following code is actually based on FLEXI MOD_VTK and trixi. jl.  Note from trixi:
        > > This order doesn't make any sense.  This is completely different from what is shown in
        > > https://blog.kitware.com/wp-content/uploads/2018/09/Source_Issue_43.pdf but this is the way it works.

        > HEXTEN :  np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (1D, tensor-product style)
        > HEXMAP : np.ndarray # MESHIO <-> IJK ordering for high-order hexahedrons (3D mapping)
    """
    if order == 1:
        map          = np.zeros((1, 1, 1), dtype=int)
        map[0, 0, 0] = 0
        tensor       = np.array([0])
        return map, tensor

    # Pre-allocate map array
    map = np.zeros((order, order, order), dtype=int)
    idx = 0

    # Principal vertices
    map[0      , 0      , 0      ] = 0
    map[order-1, 0      , 0      ] = 1
    map[order-1, order-1, 0      ] = 2
    map[0      , order-1, 0      ] = 3
    map[0      , 0      , order-1] = 4
    map[order-1, 0      , order-1] = 5
    map[order-1, order-1, order-1] = 6
    map[0      , order-1, order-1] = 7
    idx += 8

    if order > 2:
        # Number of interior nodes per edge
        nEdge = order - 2
        inner = np.arange(1, order-1)

        # Internal points of mounting edges
        map[inner  , 0      , 0      ] = np.arange(idx, idx+nEdge)  # Edge 1:  x-edge, y=0  , z=0
        idx += nEdge
        map[order-1, inner  , 0      ] = np.arange(idx, idx+nEdge)  # Edge 2:  y-edge, x=max, z=0
        idx += nEdge
        map[inner  , order-1, 0      ] = np.arange(idx, idx+nEdge)  # Edge 3:  x-edge, y=max, z=0
        idx += nEdge
        map[0      , inner  , 0      ] = np.arange(idx, idx+nEdge)  # Edge 4:  y-edge, x=0  , z=0
        idx += nEdge
        map[inner  , 0      , order-1] = np.arange(idx, idx+nEdge)  # Edge 5:  x-edge, y=0  , z=max
        idx += nEdge
        map[order-1, inner  , order-1] = np.arange(idx, idx+nEdge)  # Edge 6:  y-edge, x=max, z=max
        idx += nEdge
        map[inner  , order-1, order-1] = np.arange(idx, idx+nEdge)  # Edge 7:  x-edge, y=max, z=max
        idx += nEdge
        map[0      , inner  , order-1] = np.arange(idx, idx+nEdge)  # Edge 8:  y-edge, x=0  , z=max
        idx += nEdge
        map[0      , 0      , inner  ] = np.arange(idx, idx+nEdge)  # Edge 9:  z-edge, x=0  , y=0
        idx += nEdge
        map[order-1, 0      , inner  ] = np.arange(idx, idx+nEdge)  # Edge 10: z-edge, x=max, y=0
        idx += nEdge
        # INFO: The following two are switched compared to trixi because ParaView changed the ordering from VTK8 to VTK9 convention
        # https://gitlab.kitware.com/paraview/paraview/-/issues/20728
        map[order-1, order-1, inner  ] = np.arange(idx, idx+nEdge)  # Edge 11: z-edge, x=max, y=max
        idx += nEdge
        map[0      , order-1, inner  ] = np.arange(idx, idx+nEdge)  # Edge 12: z-edge, x=0  , y=max
        idx += nEdge

        # Number of interior nodes per face
        nFace = nEdge * nEdge

        # Faces:
        # > First index varies fastest, so emulate Fortran column-major via 'C' ravel on 'xy' meshgrid
        ii, jj = np.meshgrid(inner, inner, indexing='xy')
        iFace  = ii.ravel(order='C')
        jFace  = jj.ravel(order='C')
        del ii, jj

        map[0      , iFace, jFace] = np.arange(idx, idx+nFace)  # Face 1: x=0   (Left)   - yz plane
        idx += nFace
        map[order-1, iFace, jFace] = np.arange(idx, idx+nFace)  # Face 2: x=max (Right)  - yz plane
        idx += nFace
        map[iFace, 0      , jFace] = np.arange(idx, idx+nFace)  # Face 3: y=0   (Front)  - xz plane
        idx += nFace
        map[iFace, order-1, jFace] = np.arange(idx, idx+nFace)  # Face 4: y=max (Back)   - xz plane
        idx += nFace
        map[iFace, jFace, 0      ] = np.arange(idx, idx+nFace)  # Face 5: z=0   (Bottom) - xy plane
        idx += nFace
        map[iFace, jFace, order-1] = np.arange(idx, idx+nFace)  # Face 6: z=max (Top)    - xy plane
        idx += nFace

        # Volume
        # > Fortran order: i varies fastest, then j, then k
        nVol = nEdge ** 3
        ii, jj, kk = np.meshgrid(inner, inner, inner, indexing='ij')
        iVol = ii.ravel(order='F')
        jVol = jj.ravel(order='F')
        kVol = kk.ravel(order='F')
        del ii, jj, kk

        map[iVol, jVol, kVol] = np.arange(idx, idx+nVol)
        idx += nVol

    # Reshape into 1D array, tensor-product style
    tensor = np.zeros(order ** 3, dtype=int)
    for k in range(order):
        for j in range(order):
            for i in range(order):
                tensor[map[i, j, k]] = i + j * order + k * order * order

    return map, tensor
