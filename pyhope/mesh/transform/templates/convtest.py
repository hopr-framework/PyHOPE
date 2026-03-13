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
from __future__ import annotations
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def PostDeform(points: npt.NDArray) -> npt.NDArray:  # pragma: no cover
    """ This is the default transformation function which has to be present in every Post-Deformation template.
        PyHOPE expects this function to return the deformed points as an np.ndarray. Thus, the function signature remain unchanged.
    """

    eps: Final[float] = 1./16

    nTotal = points.shape[0]
    X_out  = np.zeros_like(points, dtype=np.float64)

    for i in range(nTotal):
        x = points[i, :]
        X_out[i, 0] = x[0] + eps * np.cos(  np.pi*(x[0]-0.5))* \
                                   np.sin(4*np.pi*(x[1]-0.5))* \
                                   np.cos(  np.pi*(x[2]-0.5))
        X_out[i, 1] = x[1] + eps * np.cos(3*np.pi*(x[0]-0.5))* \
                                   np.cos(  np.pi*(x[1]-0.5))* \
                                   np.cos(  np.pi*(x[2]-0.5))
        X_out[i, 2] = x[2] + eps * np.cos(  np.pi*(x[0]-0.5))* \
                                   np.cos(2*np.pi*(x[1]-0.5))* \
                                   np.cos(  np.pi*(x[2]-0.5))

    return points
