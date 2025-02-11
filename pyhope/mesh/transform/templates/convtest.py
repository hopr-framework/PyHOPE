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
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def PostDeform(points: np.ndarray) -> np.ndarray:
    """ This is the default transformation function which has to be present in every Post-Deformation template.
        PyHOPE expects this function to return the deformed points as an np.ndarray. Thus, the function signature remain unchanged.
    """

    eps = 1./16
    for iPoint, xPoint in enumerate(points):
        points[iPoint, 0] = xPoint[0] + eps * np.cos(  np.pi*(xPoint[0]-0.5))* \
                                              np.sin(4*np.pi*(xPoint[1]-0.5))* \
                                              np.cos(  np.pi*(xPoint[2]-0.5))
        points[iPoint, 1] = xPoint[1] + eps * np.cos(3*np.pi*(xPoint[0]-0.5))* \
                                              np.cos(  np.pi*(xPoint[1]-0.5))* \
                                              np.cos(  np.pi*(xPoint[2]-0.5))
        points[iPoint, 2] = xPoint[2] + eps * np.cos(  np.pi*(xPoint[0]-0.5))* \
                                              np.cos(2*np.pi*(xPoint[1]-0.5))* \
                                              np.cos(  np.pi*(xPoint[2]-0.5))

    return points
