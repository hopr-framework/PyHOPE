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


def PostDeform(elems: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ This is the default transformation function which has to be present in every Post-Deformation template.
        PyHOPE expects this function to return the deformed points as an np.ndarray. Thus, the function signature remain unchanged.
    """

    #  eps = 1./32
    #  list_pyram   = [points[elem.nodes] for elem in elems if elem.type % 100 == 5]
    #  points_pyram = []
    #  if len(list_pyram) > 0:
    #      points_pyram = np.concatenate(list_pyram)
    #  for iPoint, xPoint in enumerate(points):
    #      if xPoint in points_pyram:
    #          continue
    #      #  if xPoint[1] <= 1.0 and xPoint[1] >= 0.5 and xPoint[2] <= 1.0 and xPoint[2] >= 0.5:
    #      #      continue
    #
    #      points[iPoint, 0] = xPoint[0] + eps * np.cos(  np.pi*(xPoint[0]-0.5))* \
    #                                            np.sin(4*np.pi*(xPoint[1]-0.5))* \
    #                                            np.cos(  np.pi*(xPoint[2]-0.5))
    #      points[iPoint, 1] = xPoint[1] + eps * np.cos(3*np.pi*(xPoint[0]-0.5))* \
    #                                            np.cos(  np.pi*(xPoint[1]-0.5))* \
    #                                            np.cos(  np.pi*(xPoint[2]-0.5))
    #      points[iPoint, 2] = xPoint[2] + eps * np.cos(  np.pi*(xPoint[0]-0.5))* \
    #                                            np.cos(2*np.pi*(xPoint[1]-0.5))* \
    #                                            np.cos(  np.pi*(xPoint[2]-0.5))
    #
    #                                            import numpy as np

    eps = 1./32

    # 1. Identify indices of points belonging to pyramids using vectorized set logic
    pyramid_node_indices = np.unique([elem.nodes for elem in elems if elem.type % 100 == 5])

    # 2. Create a boolean mask: True for points we WANT to perturb
    mask = np.ones(len(points), dtype=bool)
    if len(pyramid_node_indices) > 0:
        mask[pyramid_node_indices] = False

    # 3. Extract only the points to be modified
    # We work on a subset to avoid unnecessary trig calculations
    target_points = points[mask]
    x = target_points[:, 0] - 0.5
    y = target_points[:, 1] - 0.5
    z = target_points[:, 2] - 0.5

    # 4. Vectorized trigonometric calculations
    # Pre-computing common terms like np.pi*x saves time
    pi_x, pi_y, pi_z = np.pi * x, np.pi * y, np.pi * z

    new_x = target_points[:, 0] + eps * np.cos(pi_x) * np.sin(4 * pi_y) #* np.cos(pi_z)
    new_y = target_points[:, 1] + eps * np.cos(3 * pi_x) * np.cos(pi_y) #* np.cos(pi_z)
    new_z = target_points[:, 2] #+ eps * np.cos(pi_x) * np.cos(2 * pi_y) * np.cos(pi_z)

    # 5. Update the original array using the mask
    points[mask, 0] = new_x
    points[mask, 1] = new_y
    points[mask, 2] = new_z

    return points
