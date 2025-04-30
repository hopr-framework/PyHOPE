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


def phill_h(x_in: float) -> float:
    xloc = x_in * 28.0
    if xloc > 54:
        xloc = 28.0 * 9.0 - xloc  # Right side of the channel

    if -1e-10 <= xloc <= 9:
        out = min(28.0, 28.0 + 6.775070969851E-03 * xloc**2 - 2.124527775800E-03 * xloc**3)
    elif 9 < xloc <= 14:
        out = 25.07355893131 + 0.9754803562315 * xloc - 0.1016116352781 * xloc**2 + 1.889794677828E-03 * xloc**3
    elif 14 < xloc <= 20:
        out = 25.79601052357 + 0.8206693007457 * xloc - 0.09055370274339 * xloc**2 + 1.626510569859E-03 * xloc**3
    elif 20 < xloc <= 30:
        out = 40.46435022819 - 1.379581654948 * xloc + 1.945884504128E-02 * xloc**2 - 2.070318932190E-04 * xloc**3
    elif 30 < xloc <= 40:
        out = 17.92461334664 + 0.8743920332081 * xloc - 0.05567361123058 * xloc**2 + 6.277731764683E-04 * xloc**3
    elif 40 < xloc <= 54:
        out = max(0.0, 56.39011190988 - 2.010520359035 * xloc + 1.644919857549E-02 * xloc**2 + 2.674976141766E-05 * xloc**3)
    else:
        out = 0.0

    return out / 28.0


def phill_normal(x_in: float) -> np.ndarray:
    xloc = x_in * 28.0
    if xloc > 54:
        xloc = 28.0 * 9.0 - xloc

    h_deriv = 0.0
    if -1e-10 <= xloc <= 9:
        h_deriv = 2 * 6.775070969851E-03 * xloc - 3 * 2.124527775800E-03 * xloc**2
    elif 9 < xloc <= 14:
        h_deriv = 0.9754803562315 - 2 * 0.1016116352781 * xloc + 3 * 1.889794677828E-03 * xloc**2
    elif 14 < xloc <= 20:
        h_deriv = 0.8206693007457 - 2 * 0.09055370274339 * xloc + 3 * 1.626510569859E-03 * xloc**2
    elif 20 < xloc <= 30:
        h_deriv = -1.379581654948 + 2 * 1.945884504128E-02 * xloc - 3 * 2.070318932190E-04 * xloc**2
    elif 30 < xloc <= 40:
        h_deriv = 0.8743920332081 - 2 * 0.05567361123058 * xloc + 3 * 6.277731764683E-04 * xloc**2
    elif 40 < xloc <= 54:
        h_deriv = -2.010520359035 + 2 * 1.644919857549E-02 * xloc + 3 * 2.674976141766E-05 * xloc**2

    if abs(h_deriv) < 1e-10:
        return np.array((0.0, 1.0))

    normal = np.array((1.0, -1.0 / h_deriv))
    return normal / np.linalg.norm(normal)


def PostDeform(points: np.ndarray) -> np.ndarray:
    """ This is the default transformation function which has to be present in every Post-Deformation template.
        PyHOPE expects this function to return the deformed points as an np.ndarray. Thus, the function signature remain unchanged.
    """

    n_total = points.shape[0]
    X_out = np.copy(points)
    h_max = 3.035

    for i in range(n_total):
        x        = points[i, :]
        xout     = np.copy(x)
        x_left   = 4.5 - abs(x[0] - 4.5)
        h        = phill_h(x_left)
        g        = 2.0 / h_max**3 * x[1]**3 - 3.0 / h_max**2 * x[1]**2 + 1.0
        xout[1] += g * h

        if x_left >= 0.1 and x[1] < h_max:
            length = xout[1] - h
            x_blend_top = 0.8
            x_blend_bottom = 1.6

            if x_left < x_blend_top:
                vec_ref_top    = phill_normal(x_blend_top)
                vec            = np.array((0.0, 1.0)) +  x_left / x_blend_top     * (vec_ref_top            - np.array((0.0, 1.0)))
            elif x_left >= x_blend_bottom:
                vec_ref_bottom = phill_normal(x_blend_bottom)
                vec            = vec_ref_bottom       + (x_left - x_blend_bottom) / (4.5 - x_blend_bottom) * (np.array((0.0, 1.0)) - vec_ref_bottom)  # noqa: E501
            else:
                vec            = phill_normal(x_left)

            if x[0] > 4.5:
                vec[0] = -vec[0]

            g = 0.9 * g**3
            xout[:2] += g * length * (vec - np.array((0.0, 1.0)))

        X_out[i, :] = xout

    return X_out
