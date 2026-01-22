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
#
# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import sys
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
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import CreateInt, CreateReal, GetInt, GetReal
    # ------------------------------------------------------

    # Readin parameters
    CreateInt( 'meshSineType', default=30 , multiple=False, help='Sine deformation type [30, 31, 32, 33, 34, 40, 41, 42, 43]')
    CreateReal('meshSineEps',               multiple=False, help='Sine deformation epsilon')
    type = GetInt( 'meshSineType')
    match type:
        case 30 | 31 | 32 | 33 | 34:
            eps = GetReal('meshSineEps', default='0.10')
        case 40 | 41 | 42 | 43:
            eps = GetReal('meshSineEps', default='0.15')
        case _:
            hopout.warning('Unknown Sine deformation type, exiting...')
            sys.exit(1)

    match type:
        case 30:  # sin[-1,1]^3
            delta = eps * np.sin(      np.pi * points[:, 0]) * np.sin(      np.pi * points[:, 1])
            points[:, :3] += delta[:, None]
        case 31:  # sin[-1,1]^3
            delta = eps * np.sin(      np.pi * points[:, 0]) * np.sin(      np.pi * points[:, 1]) * np.sin(      np.pi * points[:, 2])     # noqa: E501
            points[:, :3] += delta[:, None]
        case 32:  # sin[-1,1]^3
            delta = eps * np.sin(      np.pi * points[:, 0])
            points[:, :3] += delta[:, None]
        case 33:  # sin[-1,1]^2
            delta = eps * np.sin(      np.pi * points[:, 0]) * np.sin(      np.pi * points[:, 1])
            points[:, :2] += delta[:, None]
        case 34:  # cos3D (1.5Pi) [-1;1]^3
            delta = eps * np.cos(1.5 * np.pi * points[:, 0]) * np.cos(1.5 * np.pi * points[:, 1]) * np.cos(1.5 * np.pi * points[:, 2])     # noqa: E501
            points[:, :3] += delta[:, None]
        case 40:  # cos with coupling  [-1;1]^3 (from https://arxiv.org/pdf/1809.01178.pdf, page 20)
            x = points.copy()
            points[:, 0] = x[:, 0] + eps * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(2.0 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 2])  # noqa: E501
            points[:, 1] = x[:, 1] + eps * np.cos(1.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 2])  # noqa: E501
            points[:, 2] = x[:, 2] + eps * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(      np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 2])  # noqa: E501
        case 41:  # cos in xy with coupling  [-1;1]^2 (from https://arxiv.org/pdf/1809.01178.pdf, page 18)
            x = points.copy()
            points[:, 0] = x[:, 0] + eps * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(1.5 * np.pi * x[:, 1])
            points[:, 1] = x[:, 1] + eps * np.cos(2.0 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
        case 42:  # cos in xz with coupling  [-1;1]^2 (from https://arxiv.org/pdf/1809.01178.pdf, page 18)
            x = points.copy()
            points[:, 0] = x[:, 0] + eps * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(1.5 * np.pi * x[:, 2])
            points[:, 2] = x[:, 2] + eps * np.cos(2.0 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 2])
        case 43:  # cos in yz with coupling  [-1;1]^2 (from https://arxiv.org/pdf/1809.01178.pdf, page 18)
            x = points.copy()
            points[:, 1] = x[:, 1] + eps * np.cos(0.5 * np.pi * x[:, 1]) * np.cos(1.5 * np.pi * x[:, 2])
            points[:, 2] = x[:, 2] + eps * np.cos(2.0 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 2])

    return points
