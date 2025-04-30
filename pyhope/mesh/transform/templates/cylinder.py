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

    # TODO: Readin parameters from a configuration file
    PostDeform_R0 = 1.0       # Define appropriate values
    PostDeform_Rtorus = -1.0  # Adjust as needed
    PostDeform_Lz = 1.0       # Cylinder height scale
    PostDeform_sq = 0.0       # Spiral rotation parameter
    MeshPostDeform = 1        # Deformation mode

    nTotal = points.shape[0]
    X_out = np.zeros_like(points)

    for i in range(nTotal):
        x = points[i, :].copy()
        rr = max(abs(x[0]), abs(x[1]))

        if rr < 0.5:
            dx1 = np.array([
                0.5 * np.sqrt(2) * np.cos(0.25 * np.pi * x[1] / 0.5) - 0.5,
                0.5 * np.sqrt(2) * np.sin(0.25 * np.pi * x[1] / 0.5) - x[1]
            ])
            dx2 = np.array([
                0.5 * np.sqrt(2) * np.sin(0.25 * np.pi * x[0] / 0.5) - x[0],
                0.5 * np.sqrt(2) * np.cos(0.25 * np.pi * x[0] / 0.5) - 0.5
            ])
            alpha = 0.35
            dx = alpha * (dx1 * np.array([2 * x[0], 1.]) + dx2 * np.array([1., 2 * x[1]]))
        else:
            if abs(x[1]) < abs(x[0]):
                dx = np.array([
                    x[0] * np.sqrt(2) * np.cos(0.25 * np.pi * x[1] / x[0]) - x[0],
                    x[0] * np.sqrt(2) * np.sin(0.25 * np.pi * x[1] / x[0]) - x[1]
                ])
            else:
                dx = np.array([
                    x[1] * np.sqrt(2) * np.sin(0.25 * np.pi * x[0] / x[1]) - x[0],
                    x[1] * np.sqrt(2) * np.cos(0.25 * np.pi * x[0] / x[1]) - x[1]
                ])
            alpha = min(1., 2. * rr - 1.)
            alpha = np.sin(0.5 * np.pi * alpha)
            alpha = 1.0 * alpha + 0.35 * (1. - alpha)
            dx *= alpha

        xout = PostDeform_R0 * np.sqrt(0.5) * (x[:2] + dx)

        if MeshPostDeform == 1:
            arg = 2. * np.pi * x[2] * PostDeform_sq
        elif MeshPostDeform == 11:
            arg = 2. * np.pi * x[2] * PostDeform_sq * np.sum(xout**2)
        elif MeshPostDeform == 12:
            arg = 2. * np.pi * x[2] * PostDeform_sq * np.sum(xout**2) * (1 + 0.5 * xout[0])
        else:
            arg = 0

        rotmat = np.array([
            [np.cos(arg), -np.sin(arg)],
            [np.sin(arg),  np.cos(arg)]
        ])
        xout = np.matmul(rotmat, xout)

        if PostDeform_Rtorus < 0:
            xout = np.append(xout, x[2] * PostDeform_Lz)
        else:
            temp_z = xout[1]
            xout[1] = -(xout[0] + PostDeform_Rtorus) * np.sin(2 * np.pi * x[2])
            xout[0] = (xout[0] + PostDeform_Rtorus) * np.cos(2 * np.pi * x[2])
            xout = np.append(xout, temp_z)

        X_out[i, :] = xout

    return X_out
