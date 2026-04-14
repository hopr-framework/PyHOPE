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
        > The transformation maps a 2D square region to a cylindrical or toroidal coordinate system
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
    CreateReal( 'PostDeform_R0',      default= 1.0, multiple=False, help='Cylinder radius')                      # noqa: E251
    CreateReal( 'PostDeform_RR',      default= 0.5, multiple=False, help='Cylinder inner radius')                # noqa: E251
    CreateReal( 'PostDeform_Rtorus',  default=-1.0, multiple=False, help='z must be inside [0,1] and periodic')  # noqa: E251
    CreateReal( 'PostDeform_Lz',      default= 1.0, multiple=False)                                              # noqa: E251
    CreateReal( 'PostDeform_sq',      default= 0.0, multiple=False)                                              # noqa: E251
    # FIXME: Implement calling the other transformation types
    # CreateInt(  'MeshPostDeform',     default=1   , multiple=False, help='Deformation mode')                     # noqa: E251
    PostDeform_R0     = GetReal('PostDeform_R0')
    PostDeform_RR     = GetReal('PostDeform_RR')
    PostDeform_Rtorus = GetReal('PostDeform_Rtorus')
    PostDeform_Lz     = GetReal('PostDeform_Lz')
    PostDeform_sq     = GetReal('PostDeform_sq')
    # MeshPostDeform    = GetInt( 'MeshPostDeform')
    MeshPostDeform    = 1

    nTotal = points.shape[0]
    X_out  = np.zeros_like(points, dtype=np.float64)

    # Pre-compute constants
    sqrt2: Final[np.float64] = np.sqrt(2, dtype=np.float64)
    pFact: Final[np.float64] = PostDeform_R0 * np.sqrt(0.5, dtype=np.float64)
    Pi:    Final[np.float64] = np.pi

    # Pre-allocate arrays
    dx1 = np.empty(2, dtype=np.float64)
    dx2 = np.empty(2, dtype=np.float64)
    dx  = np.empty(2, dtype=np.float64)

    for i in range(nTotal):
        x = points[i, :].copy()
        rr = max(abs(x[0]), abs(x[1]))

        # Compute displacement based on region
        if rr < PostDeform_RR:
            # Inner region: Blending
            dx1[0] = 0.5 * sqrt2 * np.cos(0.25 * Pi * x[1] / 0.5) - 0.5
            dx1[1] = 0.5 * sqrt2 * np.sin(0.25 * Pi * x[1] / 0.5) - x[1]

            dx2[0] = 0.5 * sqrt2 * np.sin(0.25 * Pi * x[0] / 0.5) - x[0]
            dx2[1] = 0.5 * sqrt2 * np.cos(0.25 * Pi * x[0] / 0.5) - 0.5

            alpha = 0.35
            dx = alpha * (dx1 * np.array((2 * x[0], 1.)) + dx2 * np.array((1., 2 * x[1])))
        else:
            # Outer region: Single displacement field
            if abs(x[1]) < abs(x[0]):
                dx[0] = x[0] * sqrt2 * np.cos(0.25 * Pi * x[1] / x[0]) - x[0]
                dx[1] = x[0] * sqrt2 * np.sin(0.25 * Pi * x[1] / x[0]) - x[1]
            else:
                dx[0] = x[1] * sqrt2 * np.sin(0.25 * Pi * x[0] / x[1]) - x[0]
                dx[1] = x[1] * sqrt2 * np.cos(0.25 * Pi * x[0] / x[1]) - x[1]

            alpha = min(1., 2. * rr - 1.) if PostDeform_RR > 0 else 1.
            alpha = np.sin(0.5 * Pi * alpha)
            alpha = 1.0 * alpha + 0.35 * (1. - alpha)
            dx *= alpha

        # Apply base transformation
        xout = pFact * (x[:2] + dx)

        # Compute rotation angle
        match MeshPostDeform:
            case 1:
                arg = 2. * Pi * x[2] * PostDeform_sq
            case 11:
                arg = 2. * Pi * x[2] * PostDeform_sq * np.sum(xout**2)
            case 12:
                arg = 2. * Pi * x[2] * PostDeform_sq * np.sum(xout**2) * (1 + 0.5 * xout[0])
            case _:
                arg = 0

        # Apply 2D rotation
        rotmat = np.array(((np.cos(arg), -np.sin(arg)),
                          ( np.sin(arg),  np.cos(arg))))
        xout   = np.matmul(rotmat, xout)

        # Assemble final 3D transformation
        if PostDeform_Rtorus < 0:
            # Cylindrical coordinates
            X_out[i, 0] = xout[0]
            X_out[i, 1] = xout[1]
            X_out[i, 2] = x[2] * PostDeform_Lz
        else:
            # Toroidal coordinates
            temp_z      = xout[1]
            xout[1]     = -(xout[0] + PostDeform_Rtorus) * np.sin(2 * Pi * x[2])
            xout[0]     =  (xout[0] + PostDeform_Rtorus) * np.cos(2 * Pi * x[2])
            X_out[i, 0] = xout[0]
            X_out[i, 1] = xout[1]
            X_out[i, 2] = temp_z

    return X_out
