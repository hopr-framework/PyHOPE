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
    """
    This function applies a deformation transformation to the input points based on the given Fortran logic.
    The transformation maps a 2D square region to a cylindrical or toroidal coordinate system.
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import CreateReal, GetReal
    # ------------------------------------------------------

    hopout.sep()
    CreateReal('PostDeform_R0',     default= 1.0, help='Scaling factor for the cylinder/torus radius')                         # noqa: E251
    CreateReal('PostDeform_sq',     default= 0.0, help='Spiral factor along z. 0: no spiral, 1: one rotation for z in [0,1]')  # noqa: E251
    CreateReal('PostDeform_Rtorus', default=-1.0, help='>0 for torus major radius (around z); <0 for cylinder mode')           # noqa: E251
    CreateReal('PostDeform_Lz',     default= 1.0, help='Axial scaling for cylinder mode')                                      # noqa: E251

    PostDeform_R0     = GetReal('PostDeform_R0')
    PostDeform_sq     = GetReal('PostDeform_sq')
    PostDeform_Rtorus = GetReal('PostDeform_Rtorus')
    PostDeform_Lz     = GetReal('PostDeform_Lz')

    nTotal = points.shape[0]
    X_out  = np.zeros_like(points)
    Pi     = np.pi

    for i in range(nTotal):
        x  = points[i, :].copy()
        rr = max(abs(x[0]), abs(x[1]))

        if rr < 0.5:
            # inside [-0.5,0.5]^2
            # right side at x=0.5
            dx1_0 = 0.5 * np.sqrt(2.0) * np.cos(0.25 * Pi * x[1] / 0.5) - 0.5
            dx1_1 = 0.5 * np.sqrt(2.0) * np.sin(0.25 * Pi * x[1] / 0.5) - x[1]
            dx1 = np.array([dx1_0, dx1_1])

            # upper side at y=0.5
            dx2_0 = 0.5 * np.sqrt(2.0) * np.sin(0.25 * Pi * x[0] / 0.5) - x[0]
            dx2_1 = 0.5 * np.sqrt(2.0) * np.cos(0.25 * Pi * x[0] / 0.5) - 0.5
            dx2 = np.array([dx2_0, dx2_1])

            alpha = 0.35
            # coons mapping of edges, dx=0 at the corners
            dx = alpha * (dx1 * np.array([2.0 * x[0], 1.0]) + dx2 * np.array([1.0, 2.0 * x[1]]))
        else:
            # outside [-0.5,0.5]^2
            if abs(x[1]) < abs(x[0]):  # left and right quarter
                dx0 = x[0] * np.sqrt(2.0) * np.cos(0.25 * Pi * x[1] / x[0]) - x[0]
                dx1 = x[0] * np.sqrt(2.0) * np.sin(0.25 * Pi * x[1] / x[0]) - x[1]
                dx = np.array([dx0, dx1])
            else:  # upper and lower quarter (ABS(x(2)).GE.ABS(x(1)))
                dx0 = x[1] * np.sqrt(2.0) * np.sin(0.25 * Pi * x[0] / x[1]) - x[0]
                dx1 = x[1] * np.sqrt(2.0) * np.cos(0.25 * Pi * x[0] / x[1]) - x[1]
                dx = np.array([dx0, dx1])

            # maps [0.5,1] --> [0,1] and alpha=1 outside [-1,1]^2
            alpha = min(1.0, 2.0 * rr - 1.0)
            # smooth transition at the outer boundary max(|x|,|y|)=1
            alpha = np.sin(0.5 * Pi * alpha)
            # alpha=1 at max(|x|,|y|)=1, and alpha=0.35 at max(|x|,|y|)=0.5
            alpha = 1.0 * alpha + 0.35 * (1.0 - alpha)
            dx *= alpha

        xout     = np.zeros(3, dtype=points.dtype)
        xout[:2] = PostDeform_R0 * np.sqrt(0.5) * (x[:2] + dx)

        # Spiral rotation along z sq=0. no spiral, sq=1: 1 rotation along z [0,1]
        arg = 2.0 * Pi * x[2] * PostDeform_sq

        # Rotation matrix
        c = np.cos(arg)
        s = np.sin(arg)
        rotmat   = np.array([[c, -s],
                             [s,  c]], dtype=xout.dtype)
        xout[:2] = rotmat @ xout[:2]

        if PostDeform_Rtorus < 0.0:
            # cylinder
            xout[2] = x[2] * PostDeform_Lz
        else:
            # torus, z_in must be [0,1] and periodic
            # torus around z axis ,x =R*cos(phi), y=-R*sin(phi)
            # store Z
            z_val = xout[1]
            R_minor = xout[0] + PostDeform_Rtorus
            phi = 2.0 * Pi * x[2]
            xout[2] = z_val
            xout[1] = -R_minor * np.sin(phi)
            xout[0] =  R_minor * np.cos(phi)

        X_out[i, :] = xout

    return X_out
