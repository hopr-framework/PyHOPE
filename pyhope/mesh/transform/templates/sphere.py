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
    """ This function applies a deformation transformation to the input points
        > 3D box, x,y in [-1,1]^3, to sphere with radius PostDeform_R0 all points outside [-1,1]^4 will be mapped directly to a
          sphere
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.config.config as config
    from pyhope.readintools.readintools import CreateReal, GetReal, CountOption
    # ------------------------------------------------------

    # Handle the case that the variable was already used
    config.prms['meshScale']['counter'] = max(config.prms['meshScale']['counter'] - 1, 0)
    if CountOption('meshScale') and GetReal('meshScale') != 1.:
        hopout.error('"meshScale" cannot be combined with template "cylinder"')

    # Readin parameters
    CreateReal('PostDeform_R0', default=1.0, multiple=False, help='Radius of the sphere')
    PostDeform_R0 = GetReal('PostDeform_R0')

    nTotal = points.shape[0]
    X_out  = np.zeros_like(points)

    # Pre-compute constants
    sqrt3: Final[np.float64] = np.sqrt(3, dtype=np.float64)
    Pi:    Final[np.float64] = np.pi

    for i in range(nTotal):
        x = points[i, :].copy()
        rr = max(abs(x[0]), abs(x[1]), abs(x[2]))

        if rr <= 0.5:
            # Right side at x=0.5
            cosa = np.cos(0.25 * Pi * x[1] / 0.5)
            sina = np.sin(0.25 * Pi * x[1] / 0.5)
            cosb = np.cos(0.25 * Pi * x[2] / 0.5)
            sinb = np.sin(0.25 * Pi * x[2] / 0.5)
            dx1  = np.array([cosa * cosb, sina * cosb, cosa * sinb])
            dx1  = dx1 * (0.5 * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2))) - np.array((0.5, x[1], x[2]))

            # Upper side at y=0.5
            cosa = np.cos(0.25 * Pi * x[2] / 0.5)
            sina = np.sin(0.25 * Pi * x[2] / 0.5)
            cosb = np.cos(0.25 * Pi * x[0] / 0.5)
            sinb = np.sin(0.25 * Pi * x[0] / 0.5)
            dx2  = np.array([cosa * sinb, cosa * cosb, sina * cosb])
            dx2  = dx2 * (0.5 * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2))) - np.array((x[0], 0.5, x[2]))

            # Side at z=0.5
            cosa = np.cos(0.25 * Pi * x[0] / 0.5)
            sina = np.sin(0.25 * Pi * x[0] / 0.5)
            cosb = np.cos(0.25 * Pi * x[1] / 0.5)
            sinb = np.sin(0.25 * Pi * x[1] / 0.5)
            dx3  = np.array([sina * cosb, cosa * sinb, cosa * cosb])
            dx3  = dx3 * (0.5 * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2))) - np.array((x[0], x[1], 0.5))

            alpha = 0.35
            dx    = alpha * (
                   dx1 * np.array((2 * x[0], 1.0, 1.0)) +
                   dx2 * np.array((1.0, 2 * x[1], 1.0)) +
                   dx3 * np.array((1.0, 1.0, 2 * x[2]))
            )

            # Edge correction (Coons mapping) - edges: x=0.5, y=0.5
            cosa = np.sqrt(0.5)
            sina = np.sqrt(0.5)
            cosb = np.cos(0.25 * Pi * x[2] / 0.5)
            sinb = np.sin(0.25 * Pi * x[2] / 0.5)
            dx1  = np.array([cosa * cosb, sina * cosb, cosa * sinb])
            dx1  = dx1 * (0.5 * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2))) - np.array((0.5, 0.5, x[2]))

            # Edges: y=0.5, z=0.5
            cosb = np.cos(0.25 * Pi * x[0] / 0.5)
            sinb = np.sin(0.25 * Pi * x[0] / 0.5)
            dx2  = np.array([cosa * sinb, cosa * cosb, sina * cosb])
            dx2  = dx2 * (0.5 * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2))) - np.array((x[0], 0.5, 0.5))

            # Edges: x=0.5, z=0.5
            cosb = np.cos(0.25 * Pi * x[1] / 0.5)
            sinb = np.sin(0.25 * Pi * x[1] / 0.5)
            dx3  = np.array([sina * cosb, cosa * sinb, cosa * cosb])
            dx3  = dx3 * (0.5 * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2))) - np.array((0.5, x[1], 0.5))

            # Faces minus edges (Coons mapping, dx=0 at corners)
            dx -= alpha * 2.0 * (
                   dx1 * np.array((x[0],                          x[1],                          0.5 * (abs(x[0]) + abs(x[1])))) +
                   dx2 * np.array((0.5 * (abs(x[1]) + abs(x[2])), x[1],                          x[2]                         )) +
                   dx3 * np.array((x[0],                          0.5 * (abs(x[0]) + abs(x[2])), x[2]                         ))
            )

            # Apply deformation
            xout = (PostDeform_R0 / sqrt3) * (x + dx)
        else:
            # Outside [-0.5,0.5]^3, determine direction
            if abs(x[1]) < abs(x[0]) and abs(x[2]) < abs(x[0]):
                cosa = np.cos(0.25 * Pi * x[1] / x[0])
                sina = np.sin(0.25 * Pi * x[1] / x[0])
                cosb = np.cos(0.25 * Pi * x[2] / x[0])
                sinb = np.sin(0.25 * Pi * x[2] / x[0])
                dx = np.array([cosa * cosb, sina * cosb, cosa * sinb])
                dx = x[0] * dx * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2)) - x
            elif abs(x[0]) <= abs(x[1]) and abs(x[2]) < abs(x[1]):
                cosa = np.cos(0.25 * Pi * x[2] / x[1])
                sina = np.sin(0.25 * Pi * x[2] / x[1])
                cosb = np.cos(0.25 * Pi * x[0] / x[1])
                sinb = np.sin(0.25 * Pi * x[0] / x[1])
                dx = np.array([cosa * sinb, cosa * cosb, sina * cosb])
                dx = x[1] * dx * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2)) - x
            else:
                cosa = np.cos(0.25 * Pi * x[0] / x[2])
                sina = np.sin(0.25 * Pi * x[0] / x[2])
                cosb = np.cos(0.25 * Pi * x[1] / x[2])
                sinb = np.sin(0.25 * Pi * x[1] / x[2])
                dx = np.array([sina * cosb, cosa * sinb, cosa * cosb])
                dx = x[2] * dx * np.sqrt(3.0 / (cosb ** 2 + (cosa * sinb) ** 2)) - x

            alpha = min(1.0, 2.0 * rr - 1.0)
            alpha = np.sin(0.5 * Pi * alpha)
            alpha = 1.0 * alpha + 0.35 * (1.0 - alpha)
            dx *= alpha

            xout = (PostDeform_R0 / sqrt3) * (x + dx)

        X_out[i, :] = xout

    return X_out
