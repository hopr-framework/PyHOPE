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
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from typing import overload, Literal
import numpy as np
import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


# Typing helpers
@overload
def unique(a: npt.NDArray[np.float64], return_inverse: Literal[False] = False) -> npt.NDArray[np.float64]: ...
@overload
def unique(a: npt.NDArray[np.float64], return_inverse: Literal[True])          -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...               # noqa: E501
# Function
def unique(a: npt.NDArray[np.float64], return_inverse: bool = False)  -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:  # noqa: E501
    """Unique rows (axis=0) for float64 2D arrays with optional inverse mapping
    """
    if a.ndim != 2:
        raise ValueError('common_unique expects a 2D array')

    # Lexicographic row order: primary x (col 0), then y (col 1), then z (col 2)
    # np.lexsort sorts by last key first, so pass (z, y, x)
    order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
    sa    = a[order]

    # Identify starts of unique blocks
    if sa.shape[0] == 0:
        if return_inverse:
            return sa, np.empty(0, dtype=np.int64)
        return sa

    # Mask where a new unique row starts
    block     = np.empty(sa.shape[0], dtype=bool)
    block[0]  = True
    block[1:] = np.any(sa[1:] != sa[:-1], axis=1)

    unique_rows = sa[block]

    if not return_inverse:
        return unique_rows

    # Build inverse mapping: each original row -> unique row index
    # Assign group ids to sorted rows, then unsort
    groupID = np.cumsum(block) - 1
    inverse = np.empty(sa.shape[0], dtype=np.int64)
    inverse[order] = groupID

    return unique_rows, inverse

