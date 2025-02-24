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
from typing import List, Optional
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================

@cache
def HEXREORDER(order: int, incomplete: Optional[bool] = False):
    """ Converts node ordering from gmsh to meshio format
    """
    EDGEMAP = [ 0, 3, 5, 1, 8, 10, 11, 9, 2, 4, 6, 7]
    FACEMAP = [ 2, 3, 1, 4, 0, 5]

    order += 1

    if incomplete:
        nNodes = 8 + 12*(order - 2)
    else:
        nNodes = order**3

    map: List[int] = [None]*nNodes

    count = 0
    for iOrder in range(np.floor(order/2).astype(int)):
        for iNode in range(8):
            map[count] = count
            count += 1

        # Edges 
        for iEdge in range(12):
            iSlice = slice(count+(order-2*(iOrder+1))   *iEdge, count+(order-2*(iOrder+1))   *(iEdge+1))
            map[iSlice] = [count+(order-2*(iOrder+1))   *(EDGEMAP[iEdge])+iNode for iNode in range(order-2*(iOrder+1))]
        count += (order-2*(iOrder+1))*12

        # Only vertices and edges of the outermost shell required for incomplete
        if incomplete:
            return map

        # Faces
        for iFace in range(6):
            iSlice = slice(count+(order-2*(iOrder+1))**2*iFace, count+(order-2*(iOrder+1))**2*(iFace+1))
            map[iSlice] = [count+(order-2*(iOrder+1))**2*(FACEMAP[iFace])+iNode for iNode in range((order-2*(iOrder+1))**2)]
        count += (order-2*(iOrder+1))**2*6

    if order % 2 != 0:
        map[count] = count
   
    return map
